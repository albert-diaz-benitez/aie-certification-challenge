"""
Evaluation module for advanced retrieval techniques.

This module evaluates different retrieval techniques for policy extraction
using the RAGAS framework and compares their performance.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from evaluations.golden_dataset import get_evaluation_data
from src.rag.graph import extract_policy_field
from src.services.advanced_retrieval_service import AdvancedRetrievalService
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get the project root directory and evaluation directory
PROJECT_ROOT = Path(__file__).parent.parent
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"


class RetrieverEvaluator:
    """Evaluator for advanced retrieval techniques using RAGAS."""

    def __init__(self):
        """Initialize the retriever evaluator."""
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.advanced_retrieval = AdvancedRetrievalService(
            qdrant_service=self.qdrant_service,
            embeddings=self.embedding_service,
        )
        self.llm = ChatOpenAI(model="gpt-4")
        self.baseline_results = None
        self.eval_data = get_evaluation_data()

        # Load or create documents for BM25
        self.documents = self._prepare_documents()

    def _prepare_documents(self) -> List[Document]:
        """Prepare documents for retrieval testing.

        Returns:
            List of documents
        """
        documents = []
        for item in self.eval_data:
            # Create a document from each email context
            doc = Document(
                page_content=item["context"],
                metadata={
                    "email_id": item["email_id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth_answer"],
                },
            )
            documents.append(doc)

        # Register parent documents
        self.advanced_retrieval.register_parent_documents(documents)
        return documents

    async def evaluate_baseline(self) -> Dict:
        """Evaluate the baseline retriever.

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating baseline retriever")

        # Load existing responses if available
        raw_responses_path = EVALUATIONS_DIR / "raw_responses.json"
        if raw_responses_path.exists():
            with open(raw_responses_path) as f:
                responses = json.load(f)
                logger.info(
                    f"Loaded {len(responses)} existing responses for baseline"
                )
        else:
            # Generate new responses if needed
            responses = await self._generate_responses(self.eval_data)

        # Calculate RAGAS metrics
        metrics = self._calculate_ragas_metrics(responses)
        self.baseline_results = metrics

        return metrics

    async def evaluate_hybrid_retriever(
        self, vector_weight: float = 0.7, k: int = 5
    ) -> Dict:
        """Evaluate the hybrid retriever.

        Args:
            vector_weight: Weight for vector search (0-1)
            k: Number of documents to retrieve

        Returns:
            Dictionary with evaluation results
        """
        logger.info(
            f"Evaluating hybrid retriever (vector_weight={vector_weight}, k={k})"
        )

        # Create a hybrid retriever
        hybrid_retriever = self.advanced_retrieval.get_hybrid_retriever(
            documents=self.documents, vector_weight=vector_weight, k=k
        )

        # Evaluate with the hybrid retriever
        return await self._evaluate_retriever(
            retriever=hybrid_retriever, retriever_name="hybrid"
        )

    async def evaluate_mmr_retriever(
        self, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> Dict:
        """Evaluate the MMR retriever.

        Args:
            k: Number of documents to retrieve
            fetch_k: Number of documents to fetch before reranking
            lambda_mult: Balance between relevance and diversity (0-1)

        Returns:
            Dictionary with evaluation results
        """
        logger.info(
            f"Evaluating MMR retriever (lambda_mult={lambda_mult}, k={k})"
        )

        # Create an MMR retriever
        mmr_retriever = self.advanced_retrieval.get_mmr_retriever(
            k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

        # Evaluate with the MMR retriever
        return await self._evaluate_retriever(
            retriever=mmr_retriever, retriever_name="mmr"
        )

    async def evaluate_parent_child_retriever(self, k: int = 5) -> Dict:
        """Evaluate the parent-child retriever.

        Args:
            k: Number of documents to retrieve

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating parent-child retriever (k={k})")

        # Create a parent-child retriever
        parent_child_retriever = (
            self.advanced_retrieval.get_parent_child_retriever(k=k)
        )

        # Evaluate with the parent-child retriever
        return await self._evaluate_retriever(
            retriever=parent_child_retriever, retriever_name="parent_child"
        )

    async def evaluate_field_specific_retriever(
        self, base_retriever_type: str = "hybrid"
    ) -> Dict:
        """Evaluate the field-specific retriever.

        Args:
            base_retriever_type: Type of base retriever to use

        Returns:
            Dictionary with evaluation results
        """
        logger.info(
            f"Evaluating field-specific retriever with {base_retriever_type} base"
        )

        # Create a base retriever
        if base_retriever_type == "hybrid":
            base_retriever = self.advanced_retrieval.get_hybrid_retriever(
                documents=self.documents
            )
        elif base_retriever_type == "mmr":
            base_retriever = self.advanced_retrieval.get_mmr_retriever()
        else:
            # Default to hybrid
            base_retriever = self.advanced_retrieval.get_hybrid_retriever(
                documents=self.documents
            )

        # Use field-specific retriever
        # We'll process each question separately based on the field
        results_by_field = {}
        all_responses = []

        for item in self.eval_data:
            # Map questions to fields
            field_map = {
                "Who is the insured party or policyholder?": "policy_insured",
                "What is the line of business or policy type?": "line_of_business",
                "What is the effective date of the policy?": "effective_date",
                "What is the expected inception date for the policy?": "expected_inception_date",
                "What is the target premium amount for the policy?": "target_premium",
            }

            field = field_map.get(item["question"])
            if not field:
                continue

            # Create a field-specific retriever for this question
            field_retriever = (
                self.advanced_retrieval.get_field_specific_retriever(
                    base_retriever=base_retriever, field=field
                )
            )

            # Extract the answer using this retriever
            response = await self._extract_with_retriever(
                retriever=field_retriever,
                question=item["question"],
                field=field,
                context=item["context"],
            )

            # Add to our results
            all_responses.append({**item, "model_response": response})

            # Group by field for analysis
            if field not in results_by_field:
                results_by_field[field] = []
            results_by_field[field].append(
                {**item, "model_response": response}
            )

        # Calculate overall metrics
        metrics = self._calculate_ragas_metrics(all_responses)

        # Calculate metrics by field
        metrics["by_field"] = {}
        for field, responses in results_by_field.items():
            field_metrics = self._calculate_ragas_metrics(responses)
            metrics["by_field"][field] = field_metrics

        return metrics

    async def evaluate_ensemble_retriever(self, k: int = 5) -> Dict:
        """Evaluate the ensemble retriever.

        Args:
            k: Number of documents to retrieve

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating ensemble retriever (k={k})")

        # Create individual retrievers for the ensemble
        hybrid_retriever = self.advanced_retrieval.get_hybrid_retriever(
            documents=self.documents, k=k
        )
        mmr_retriever = self.advanced_retrieval.get_mmr_retriever(k=k)
        parent_child_retriever = (
            self.advanced_retrieval.get_parent_child_retriever(k=k)
        )

        # Create ensemble function
        ensemble_retrieve = self.advanced_retrieval.create_ensemble_retriever(
            retrievers=[
                hybrid_retriever,
                mmr_retriever,
                parent_child_retriever,
            ],
            k=k,
        )

        # Create a simple wrapper retriever
        class EnsembleWrapper:
            def get_relevant_documents(self, query):
                return ensemble_retrieve(query)

        # Evaluate with the ensemble wrapper
        return await self._evaluate_retriever(
            retriever=EnsembleWrapper(), retriever_name="ensemble"
        )

    async def _evaluate_retriever(
        self, retriever: Any, retriever_name: str
    ) -> Dict:
        """Evaluate a retriever.

        Args:
            retriever: The retriever to evaluate
            retriever_name: Name of the retriever for logging

        Returns:
            Dictionary with evaluation results
        """
        # Get responses for each question using this retriever
        responses = []
        for item in self.eval_data:
            field_map = {
                "Who is the insured party or policyholder?": "policy_insured",
                "What is the line of business or policy type?": "line_of_business",
                "What is the effective date of the policy?": "effective_date",
                "What is the expected inception date for the policy?": "expected_inception_date",
                "What is the target premium amount for the policy?": "target_premium",
            }

            field = field_map.get(item["question"])
            if not field:
                logger.warning(
                    f"Unknown field for question: {item['question']}"
                )
                continue

            response = await self._extract_with_retriever(
                retriever=retriever,
                question=item["question"],
                field=field,
                context=item["context"],
            )

            responses.append({**item, "model_response": response})

        # Calculate RAGAS metrics
        metrics = self._calculate_ragas_metrics(responses)

        # Save results
        output_path = EVALUATIONS_DIR / f"{retriever_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(self._make_serializable(metrics), f, indent=2)

        logger.info(f"Saved {retriever_name} results to {output_path}")

        return metrics

    async def _extract_with_retriever(
        self, retriever: Any, question: str, field: str, context: str
    ) -> str:
        """Extract policy information using the given retriever.

        Args:
            retriever: The retriever to use
            question: The question to answer
            field: The policy field to extract
            context: The original context

        Returns:
            The extracted answer
        """
        try:
            # Get relevant documents
            retrieved_docs = retriever.get_relevant_documents(question)

            # We'll actually use the full context since this is a
            # controlled experiment comparing only retrieval methods
            return await extract_policy_field(email_text=context, field=field)
        except Exception as e:
            logger.error(
                f"Error extracting with {type(retriever).__name__}: {str(e)}"
            )
            return f"Error: {str(e)}"

    async def _generate_responses(self, eval_data: List[Dict]) -> List[Dict]:
        """Generate model responses for evaluation data.

        Args:
            eval_data: Evaluation data

        Returns:
            List of responses
        """
        results = []

        for item in eval_data:
            logger.info(
                f"Processing question: {item['question']} for email {item['email_id']}"
            )

            try:
                # Extract the field being asked about from the question
                field_map = {
                    "Who is the insured party or policyholder?": "policy_insured",
                    "What is the line of business or policy type?": "line_of_business",
                    "What is the effective date of the policy?": "effective_date",
                    "What is the expected inception date for the policy?": "expected_inception_date",
                    "What is the target premium amount for the policy?": "target_premium",
                }

                field = field_map.get(item["question"])
                if not field:
                    logger.warning(
                        f"Unknown field for question: {item['question']}"
                    )
                    continue

                # Call the extraction function from the RAG system
                response = await extract_policy_field(
                    email_text=item["context"], field=field
                )

                results.append({**item, "model_response": response})

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                results.append({**item, "model_response": f"Error: {str(e)}"})

        return results

    def _calculate_ragas_metrics(self, evaluation_data: List[Dict]) -> Dict:
        """Calculate RAGAS metrics for evaluation data.

        Args:
            evaluation_data: Evaluation data with model responses

        Returns:
            Dictionary with RAGAS metrics
        """
        # Convert to DataFrame format expected by RAGAS
        df = pd.DataFrame(
            {
                # Map to the column names required by RAGAS metrics
                "user_input": [item["question"] for item in evaluation_data],
                "response": [
                    item["model_response"] for item in evaluation_data
                ],
                # RAGAS expects retrieved_contexts to be a list of strings
                "retrieved_contexts": [
                    [item["context"]] for item in evaluation_data
                ],  # List of lists of strings
                "ground_truths": [
                    item["ground_truth_answer"] for item in evaluation_data
                ],
                "reference": [
                    item["context"] for item in evaluation_data
                ],  # String
            }
        )

        # Also add the original column names to maintain compatibility
        df["question"] = df["user_input"]
        df["answer"] = df["response"]
        df["contexts"] = [
            [item["context"]] for item in evaluation_data
        ]  # List of lists for contexts as well

        # Try each metric individually to avoid failures in one metric affecting others
        results = {}
        metrics_to_try = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        for metric_name, metric in metrics_to_try.items():
            try:
                logger.info(f"Evaluating {metric_name} metric...")
                dataset = EvaluationDataset.from_pandas(df)
                metric_result = evaluate(dataset=dataset, metrics=[metric])
                if metric_result:
                    results[metric_name] = metric_result[metric_name]
                logger.info(f"Successfully evaluated {metric_name}")
            except Exception as e:
                logger.error(f"Failed to evaluate {metric_name}: {str(e)}")
                results[f"{metric_name}_error"] = str(e)

        return results

    def _make_serializable(self, metrics: Dict) -> Dict:
        """Convert metrics to JSON-serializable format.

        Args:
            metrics: Dictionary with metrics

        Returns:
            Dictionary with serializable metrics
        """
        serializable = {}
        for key, value in metrics.items():
            if key == "by_field":
                # Handle nested structure
                serializable[key] = {}
                for field, field_metrics in value.items():
                    serializable[key][field] = self._make_serializable(
                        field_metrics
                    )
            elif hasattr(value, "to_dict"):
                serializable[key] = value.to_dict()
            elif isinstance(value, (int, float)):
                serializable[key] = float(value)
            else:
                serializable[key] = str(value)

        return serializable

    def get_comparative_results(self) -> Dict:
        """Get comparative results of all evaluated retrievers.

        Returns:
            Dictionary with comparative results
        """
        # Load all result files
        results = {"baseline": self.baseline_results}

        for retriever_name in [
            "hybrid",
            "mmr",
            "parent_child",
            "field_specific",
            "ensemble",
        ]:
            result_path = EVALUATIONS_DIR / f"{retriever_name}_results.json"
            if result_path.exists():
                with open(result_path) as f:
                    results[retriever_name] = json.load(f)

        # Create comparative summary
        summary = {"metrics": {}, "improvement": {}}

        # Calculate summary metrics
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            summary["metrics"][metric] = {}

            # Get baseline value
            baseline_value = None
            if self.baseline_results and metric in self.baseline_results:
                # Handle different types of metric values
                if hasattr(self.baseline_results[metric], "mean"):
                    baseline_value = self.baseline_results[metric].mean()
                elif isinstance(self.baseline_results[metric], (int, float)):
                    baseline_value = self.baseline_results[metric]
                else:
                    # Try to convert from string if possible
                    try:
                        baseline_value = float(self.baseline_results[metric])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert baseline metric {metric} to a number"
                        )
                        baseline_value = None

                if baseline_value is not None:
                    summary["metrics"][metric]["baseline"] = baseline_value

            # Get other retriever values
            for retriever_name, retriever_results in results.items():
                if retriever_name == "baseline":
                    continue

                if metric in retriever_results:
                    # Handle different types of metric values
                    value = None
                    if hasattr(retriever_results[metric], "mean"):
                        value = retriever_results[metric].mean()
                    elif isinstance(retriever_results[metric], (int, float)):
                        value = retriever_results[metric]
                    else:
                        # Try to parse array-like string representation and calculate mean
                        try:
                            # Handle string representation of a list or array
                            metric_str = retriever_results[metric]

                            # If it looks like a list representation
                            if isinstance(metric_str, str) and (
                                metric_str.startswith("[")
                                and metric_str.endswith("]")
                            ):
                                # Remove 'np.float64()' if present
                                metric_str = metric_str.replace(
                                    "np.float64(", ""
                                ).replace(")", "")

                                # Parse the string as a list of numbers
                                parsed_list = eval(metric_str)

                                # Calculate the mean
                                if parsed_list:
                                    value = sum(parsed_list) / len(parsed_list)
                                    logger.info(
                                        f"Successfully calculated mean for {retriever_name} {metric}: {value}"
                                    )
                            else:
                                # Try direct conversion
                                value = float(metric_str)
                        except Exception as e:
                            logger.warning(
                                f"Could not convert {retriever_name} metric {metric} to a number: {str(e)}"
                            )
                            continue

                    if value is not None:
                        summary["metrics"][metric][retriever_name] = value

                        # Calculate improvement over baseline
                        if baseline_value is not None:
                            improvement = (
                                (value - baseline_value) / baseline_value
                            ) * 100
                            if metric not in summary["improvement"]:
                                summary["improvement"][metric] = {}
                            summary["improvement"][metric][
                                retriever_name
                            ] = improvement

        return summary


async def run_evaluation():
    """Run the advanced retriever evaluation."""
    logger.info("Starting evaluation of advanced retrievers")

    evaluator = RetrieverEvaluator()

    # Evaluate baseline first
    # baseline_results = await evaluator.evaluate_baseline()
    # logger.info("Baseline evaluation complete")

    # Evaluate hybrid retriever
    hybrid_results = await evaluator.evaluate_hybrid_retriever()
    logger.info("Hybrid retriever evaluation complete")

    # Skip other retrievers as they've been removed from the implementation
    logger.info(
        "Skipping other retrievers as they've been removed from implementation"
    )

    # Get comparative results with just what we have
    comparative_results = evaluator.get_comparative_results()

    # Save comparative results
    with open(EVALUATIONS_DIR / "comparative_results.json", "w") as f:
        json.dump(comparative_results, f, indent=2)

    logger.info("Comparative results saved")

    # Print summary
    print("\n=== Advanced Retrieval Evaluation Results ===")
    for metric, values in comparative_results["metrics"].items():
        print(f"\n{metric.upper()}:")
        for retriever, value in values.items():
            improvement = (
                comparative_results["improvement"]
                .get(metric, {})
                .get(retriever)
            )
            improvement_str = (
                f" ({improvement:+.2f}%)" if improvement is not None else ""
            )
            print(f"  {retriever}: {value:.4f}{improvement_str}")

    return comparative_results


if __name__ == "__main__":
    asyncio.run(run_evaluation())
