import uuid
from datetime import datetime, timedelta

import pytest
from qdrant_client.http import models as qmodels

from src.models.email_models import Email, EmailMetadata, VectorizedEmail
from src.services.qdrant_service import QdrantService


class TestQdrantService:
    """Integration tests for the QdrantService"""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up before each test"""
        self.qdrant_service = QdrantService()

        # Create a unique collection name for this test
        self.test_collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        self.original_collection_name = self.qdrant_service.collection_name

        # Use the test collection for this test
        self.qdrant_service.collection_name = self.test_collection_name

        # Ensure the test collection exists
        self.qdrant_service._ensure_collection_exists()

        yield

        # Cleanup: Delete the test collection after the test
        try:
            self.qdrant_service.client.delete_collection(
                self.test_collection_name
            )
        except Exception as e:
            print(f"Error cleaning up test collection: {str(e)}")

        # Restore the original collection name
        self.qdrant_service.collection_name = self.original_collection_name

    def test_collection_creation(self):
        """Test that the collection is created with correct schema"""

        # Check that the collection exists
        collections = self.qdrant_service.client.get_collections().collections
        collection_names = [c.name for c in collections]

        assert self.qdrant_service.collection_name in collection_names

        # Check collection schema
        collection_info = self.qdrant_service.client.get_collection(
            collection_name=self.qdrant_service.collection_name
        )

        assert (
            self.qdrant_service.vector_size
            == collection_info.config.params.vectors.size
        )
        assert (
            qmodels.Distance.COSINE
            == collection_info.config.params.vectors.distance
        )

    def test_store_and_query_email_vector(self):
        """Test storing a vector and then querying it"""
        # Create test vector email
        vector_id = str(uuid.uuid4())
        vector_size = self.qdrant_service.vector_size
        test_vector = [0.1] * vector_size

        # Test dates
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Create test email metadata
        metadata = EmailMetadata(
            email_id="test123",
            subject="Test Subject",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            date_received=now,
            date_processed=now,
            has_attachments=False,
            importance="normal",
        )

        # Create email
        email = Email(
            metadata=metadata,
            body_text="This is a test email body",
            body_html="<html><body>This is a test email body</body></html>",
        )

        # Create vectorized email
        vectorized_email = VectorizedEmail(
            email=email,
            vector=test_vector,
            vector_id=vector_id,
            collection_name=self.qdrant_service.collection_name,
            embedding_model="test-model",
            date_embedded=now,
        )

        # Store the vector
        stored_id = self.qdrant_service.store_email_vector(vectorized_email)
        assert stored_id == vector_id

        # Query within the same time range (should find our email)
        results = self.qdrant_service.query_by_date_range(
            start_date=yesterday, end_date=tomorrow
        )

        assert len(results) == 1
        assert results[0]["email_id"] == "test123"
        assert results[0]["subject"] == "Test Subject"
        assert results[0]["sender"] == "test@example.com"

        # Query outside of time range (should not find our email)
        earlier_results = self.qdrant_service.query_by_date_range(
            start_date=yesterday - timedelta(days=10),
            end_date=yesterday - timedelta(days=1),
        )

        assert len(earlier_results) == 0
