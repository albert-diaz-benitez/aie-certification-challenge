"""
Advanced Retrieval Service for policy extraction.

This module implements the hybrid retrieval technique to improve
the context relevance and precision for policy information extraction.
"""

import logging
from typing import Any, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from src.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorStoreAdapter(VectorStore):
    """Adapter class that wraps QdrantService to implement LangChain's VectorStore interface."""

    def __init__(self, qdrant_service: QdrantService):
        """Initialize the adapter with a QdrantService instance.

        Args:
            qdrant_service: The QdrantService to wrap
        """
        self.qdrant_service = qdrant_service
        self._embeddings = None

    @property
    def embeddings(self):
        """Get the embeddings model.

        Returns:
            The embeddings model
        """
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        """Set the embeddings model.

        Args:
            value: The embeddings model to use
        """
        self._embeddings = value

    def similarity_search(
        self, query: str, k: int = 5, **kwargs
    ) -> List[Document]:
        """Perform a similarity search against the vector store.

        Args:
            query: The query text
            k: Number of results to return

        Returns:
            List of Documents most similar to the query
        """
        # Use the embeddings model to generate a vector for the query
        if hasattr(self, "embeddings") and self.embeddings:
            query_vector = self.embeddings.embed_query(query)

            # Use the qdrant_service's client directly to perform the search
            search_result = self.qdrant_service.client.search(
                collection_name=self.qdrant_service.collection_name,
                query_vector=query_vector,
                limit=k,
            )

            # Convert search results to Documents
            documents = []
            for result in search_result:
                # Extract content and metadata from the search result
                payload = result.payload or {}
                score = result.score

                # Use the body text as content
                content = payload.get("body_text", "")

                # Collect remaining payload as metadata, including the score
                metadata = {
                    "score": score,
                    "id": str(result.id),
                    **{k: v for k, v in payload.items() if k != "body_text"},
                }

                documents.append(
                    Document(page_content=content, metadata=metadata)
                )

            return documents

        else:
            # If no embeddings model is available, return empty results
            logger.warning(
                "No embeddings model available for similarity search"
            )
            return []

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs,
    ) -> List[Document]:
        """Perform a search using Maximum Marginal Relevance reranking.

        Args:
            query: The query text
            k: Number of results to return
            fetch_k: Number of documents to fetch before reranking
            lambda_mult: Balance between relevance and diversity (0-1)

        Returns:
            List of Documents after MMR reranking
        """
        # Simplified implementation - fetch more results and do basic filtering
        # In a real implementation, you would implement proper MMR reranking
        docs = self.similarity_search(query, fetch_k, **kwargs)

        # Simple deduplication based on content
        seen_contents = set()
        unique_docs = []

        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
                if len(unique_docs) >= k:
                    break

        return unique_docs

    @classmethod
    def from_texts(cls, *args, **kwargs):
        """Required method for VectorStore, but not needed for this adapter."""
        raise NotImplementedError(
            "QdrantVectorStoreAdapter does not support creating from texts directly."
        )


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines dense and sparse retrieval methods.

    This retriever combines BM25 (keyword-based) and vector similarity search
    to provide improved retrieval for insurance policy information.
    """

    # Define fields as class variables for Pydantic
    vector_store: VectorStore
    bm25_retriever: BM25Retriever
    vector_weight: float = 0.7  # Optimized for better vector search weight
    bm25_weight: float = 0.3
    k: int = 5

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_documents: List[Document],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 5,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: The vector store for dense retrieval
            bm25_documents: The documents to use for BM25 retrieval
            vector_weight: Weight for the vector similarity scores (0-1)
            bm25_weight: Weight for the BM25 scores (0-1)
            k: Number of documents to retrieve
        """
        # Create BM25 retriever from documents
        bm25_retriever = BM25Retriever.from_documents(bm25_documents)
        bm25_retriever.k = k

        # Call parent class's __init__ with all required fields
        super().__init__(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k=k,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """Get relevant documents using both dense and sparse retrieval.

        Args:
            query: The query to search for

        Returns:
            List of relevant documents
        """
        # Get results from both retrievers
        sparse_docs = self.bm25_retriever.get_relevant_documents(query)
        vector_docs = self.vector_store.similarity_search(query, self.k)

        # Create a dictionary to store combined scores
        all_docs = {}

        # Process BM25 results - normalize scores
        if sparse_docs:
            max_bm25_score = max(
                doc.metadata.get("score", 0) for doc in sparse_docs
            )
            for i, doc in enumerate(sparse_docs):
                score = doc.metadata.get("score", 0)
                if max_bm25_score > 0:
                    normalized_score = score / max_bm25_score
                else:
                    normalized_score = 0

                doc_id = doc.metadata.get("id", f"sparse_{i}")
                all_docs[doc_id] = {
                    "doc": doc,
                    "sparse_score": normalized_score * self.bm25_weight,
                    "dense_score": 0,
                }

        # Process vector results - normalize scores
        if vector_docs:
            max_vector_score = max(
                doc.metadata.get("score", 0) for doc in vector_docs
            )
            for i, doc in enumerate(vector_docs):
                score = doc.metadata.get("score", 0)
                if max_vector_score > 0:
                    normalized_score = score / max_vector_score
                else:
                    normalized_score = 0

                doc_id = doc.metadata.get("id", f"dense_{i}")
                if doc_id in all_docs:
                    all_docs[doc_id]["dense_score"] = (
                        normalized_score * self.vector_weight
                    )
                else:
                    all_docs[doc_id] = {
                        "doc": doc,
                        "sparse_score": 0,
                        "dense_score": normalized_score * self.vector_weight,
                    }

        # Calculate combined scores and rank documents
        for doc_info in all_docs.values():
            doc_info["combined_score"] = (
                doc_info["sparse_score"] + doc_info["dense_score"]
            )

        # Sort by combined score and take top k
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        # Return top k documents
        results = [doc_info["doc"] for doc_info in sorted_docs[: self.k]]
        return results


class AdvancedRetrievalService:
    """Service for hybrid retrieval technique in policy extraction."""

    def __init__(self, qdrant_service: QdrantService, embeddings: Any):
        """Initialize the advanced retrieval service.

        Args:
            qdrant_service: The Qdrant service for vector search
            embeddings: The embeddings model or service
        """
        self.qdrant_service = qdrant_service
        # Wrap the QdrantService with our adapter
        self.vector_store_adapter = QdrantVectorStoreAdapter(qdrant_service)
        # If we're passed an EmbeddingService, use OpenAI embeddings directly
        # This ensures compatibility with methods that expect an Embeddings object
        self.embeddings = (
            OpenAIEmbeddings()
            if hasattr(embeddings, "create_embedding")
            else embeddings
        )
        # Add embeddings to the adapter
        self.vector_store_adapter.embeddings = self.embeddings
        self.parent_documents = {}  # For storing email context if needed

    def get_hybrid_retriever(
        self, documents: List[Document], vector_weight: float = 0.7, k: int = 5
    ) -> HybridRetriever:
        """Create a hybrid retriever (dense + sparse).

        Args:
            documents: Documents to use for BM25
            vector_weight: Weight for vector search (0-1)
            k: Number of documents to retrieve

        Returns:
            A hybrid retriever
        """
        return HybridRetriever(
            vector_store=self.vector_store_adapter,  # Use the adapter instead of the service directly
            bm25_documents=documents,
            vector_weight=vector_weight,
            bm25_weight=1.0 - vector_weight,
            k=k,
        )

    def register_parent_documents(self, documents: List[Document]) -> None:
        """Register parent documents for retrieval.

        Args:
            documents: The parent documents
        """
        for doc in documents:
            doc_id = doc.metadata.get("email_id") or doc.metadata.get("id")
            if doc_id:
                self.parent_documents[doc_id] = doc
