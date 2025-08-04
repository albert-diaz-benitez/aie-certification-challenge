import os
from datetime import datetime

import pytest

from src.models.email_models import Email, EmailAttachment, EmailMetadata
from src.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Integration tests for the EmbeddingService"""

    @pytest.fixture
    def sample_email(self):
        """Create a sample email for testing"""
        # Create metadata
        metadata = EmailMetadata(
            email_id="test123",
            subject="Test Email Subject",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            cc=["cc@example.com"],
            date_received=datetime.now(),
            date_processed=datetime.now(),
            has_attachments=True,
            importance="high",
        )

        # Create attachments
        attachment = EmailAttachment(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            size=12,
        )

        # Create email
        email = Email(
            metadata=metadata,
            body_text="This is a test email body with some content.",
            body_html="<html><body>This is a test email body with some content.</body></html>",
            attachments=[attachment],
        )

        return email

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OpenAI API key not available",
    )
    async def test_live_embedding_creation(self, sample_email):
        """Test creating embeddings with the actual OpenAI API (only runs if API key is available)"""
        embedding_service = EmbeddingService()

        # Prepare the email text
        email_text = embedding_service.prepare_email_text(sample_email)

        # Create the embedding
        embedding = await embedding_service.create_embedding(email_text)

        # Verify the result structure (exact values will depend on the API response)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
