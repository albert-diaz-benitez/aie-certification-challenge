"""
Evaluation module for the RAG system using RAGAS metrics.

This module evaluates the policy extraction RAG system against a silver dataset
using key RAGAS metrics: faithfulness, response relevance, context precision,
and context recall.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from evaluations.golden_dataset import get_evaluation_data
from src.rag.graph import extract_policy_field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Get the project root directory and evaluation directory
PROJECT_ROOT = Path(__file__).parent.parent
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"


async def generate_model_responses(eval_data: List[Dict]) -> List[Dict]:
    """
    Generate model responses for each question in the evaluation data.

    Args:
        eval_data: List of dictionaries containing questions and contexts

    Returns:
        Updated list with model responses added
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
            # Include failed responses with error message
            results.append({**item, "model_response": f"Error: {str(e)}"})

    return results


def calculate_ragas_metrics(evaluation_data: List[Dict]) -> Dict:
    """
    Calculate RAGAS metrics for the evaluation data.

    Args:
        evaluation_data: Evaluation data with model responses

    Returns:
        Dictionary with RAGAS metrics results
    """
    # Convert to DataFrame format expected by RAGAS
    df = pd.DataFrame(
        {
            # Map to the column names required by RAGAS metrics
            "user_input": [item["question"] for item in evaluation_data],
            "response": [item["model_response"] for item in evaluation_data],
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

    # For debugging - print column names and data types
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Reference type: {type(df['reference'].iloc[0])}")
    logger.info(
        f"Retrieved contexts type: {type(df['retrieved_contexts'].iloc[0])} with value: {df['retrieved_contexts'].iloc[0]}"
    )

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

    if not results:
        results["error"] = "All metrics failed to evaluate"

    return results


def save_results(results: Dict, filename: str = "ragas_results.json") -> None:
    """
    Save evaluation results to file.

    Args:
        results: Dictionary with evaluation results
        filename: Output filename
    """
    # Create evaluations directory if it doesn't exist
    EVALUATIONS_DIR.mkdir(exist_ok=True)

    output_path = EVALUATIONS_DIR / filename

    # Convert to serializable format
    serializable_results = {}
    for metric, value in results.items():
        if hasattr(value, "to_dict"):
            serializable_results[metric] = value.to_dict()
        elif isinstance(value, (int, float)):
            serializable_results[metric] = float(value)
        else:
            # For error messages or other string values, just store as is
            serializable_results[metric] = str(value)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


async def run_evaluation():
    """Run the full evaluation process."""
    logger.info("Starting RAG system evaluation with RAGAS")

    # Define the path for raw responses
    raw_responses_path = EVALUATIONS_DIR / "raw_responses.json"

    # Get evaluation data
    eval_data = get_evaluation_data()
    logger.info(f"Loaded {len(eval_data)} evaluation samples")

    # Check if raw responses already exist
    responses = None
    if (raw_responses_path).exists():
        try:
            logger.info(
                f"Found existing raw responses at {raw_responses_path}"
            )
            with open(raw_responses_path) as f:
                responses = json.load(f)
            logger.info(f"Loaded {len(responses)} existing responses")

            # Simple validation to ensure responses match our evaluation data
            if len(responses) != len(eval_data):
                logger.warning(
                    f"Number of existing responses ({len(responses)}) does not match evaluation data ({len(eval_data)})"
                )
                responses = None
        except Exception as e:
            logger.warning(
                f"Error loading existing responses: {e}. Will generate new responses."
            )
            responses = None

    # Generate responses only if we don't have valid ones already
    if responses is None:
        logger.info("Generating new responses...")
        # Generate responses using the RAG system
        responses = await generate_model_responses(eval_data)
        logger.info(f"Generated {len(responses)} responses")

        # Make sure directory exists
        EVALUATIONS_DIR.mkdir(exist_ok=True)

        # Save raw responses for future runs
        with open(raw_responses_path, "w") as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Saved raw responses to {raw_responses_path}")
    else:
        logger.info("Using existing responses instead of calling OpenAI API")

    # Calculate RAGAS metrics
    metrics = calculate_ragas_metrics(responses)
    logger.info("Calculated RAGAS metrics")

    # Save results
    save_results(metrics)

    # Print summary
    print("\n=== RAGAS Evaluation Results ===")
    for metric, value in metrics.items():
        if hasattr(value, "mean"):
            print(f"{metric}: {value.mean()}")
        else:
            print(f"{metric}: {value}")

    return metrics


if __name__ == "__main__":
    asyncio.run(run_evaluation())
