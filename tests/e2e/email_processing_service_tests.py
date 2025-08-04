import uuid
from datetime import datetime, timedelta

import pytest

from src.services.email_processing_service import EmailProcessingService


class TestEmailProcessingService:
    """End-to-end tests for the EmailProcessingService"""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up the test environment with mocks"""
        # Create the service instance
        self.email_processing_service = EmailProcessingService()

        # Create a unique collection name for this test
        self.test_collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        self.original_collection_name = (
            self.email_processing_service.qdrant_service.collection_name
        )

        # Use the test collection for this test
        self.email_processing_service.qdrant_service.collection_name = (
            self.test_collection_name
        )

        # Ensure the test collection exists
        self.email_processing_service.qdrant_service._ensure_collection_exists()

        yield

        # Cleanup: Delete the test collection after the test
        try:
            self.email_processing_service.qdrant_service.client.delete_collection(
                self.test_collection_name
            )
        except Exception as e:
            print(f"Error cleaning up test collection: {str(e)}")

        # Restore the original collection name
        self.email_processing_service.qdrant_service.collection_name = (
            self.original_collection_name
        )

    @pytest.mark.asyncio
    async def test_process_emails_end_to_end(self):
        """Test the full email processing workflow end-to-end:"""
        # Run the email processing
        await self.email_processing_service.process_emails()

        # Test dates
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        vectors = (
            self.email_processing_service.qdrant_service.query_by_date_range(
                end_date=tomorrow, start_date=yesterday
            )
        )

        assert len(vectors) > 0
