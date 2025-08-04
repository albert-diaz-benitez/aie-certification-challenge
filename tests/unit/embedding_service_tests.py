from datetime import datetime
from unittest import mock

import pytest

from src.models.email_models import Email, EmailAttachment, EmailMetadata
from src.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Unit tests for the EmbeddingService"""

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response with a test embedding"""

        # Create a mock embedding response
        class MockEmbeddingData:
            def __init__(self):
                self.embedding = [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ] * 10  # 50-dimensional vector

        class MockEmbeddingResponse:
            def __init__(self):
                self.data = [MockEmbeddingData()]

        return MockEmbeddingResponse()

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

    def test_prepare_email_text(self, sample_email):
        """Test that email text is properly prepared for embedding"""
        embedding_service = EmbeddingService(model_name="test-model")

        # Get the prepared text
        result = embedding_service.prepare_email_text(sample_email)

        # Check that the result contains key information
        assert "Subject: Test Email Subject" in result
        assert "From: sender@example.com" in result
        assert "To: recipient@example.com" in result
        assert "CC: cc@example.com" in result
        assert "Content:" in result
        assert "This is a test email body with some content." in result
        assert "Attachments:" in result
        assert "test.pdf" in result

    def test_prepare_email_text_without_cc_and_attachments(self, sample_email):
        """Test text preparation without CC or attachments"""
        # Modify the email to not have CC or attachments
        sample_email.metadata.cc = []
        sample_email.attachments = []

        embedding_service = EmbeddingService(model_name="test-model")

        # Get the prepared text
        result = embedding_service.prepare_email_text(sample_email)

        # Check content
        assert "Subject: Test Email Subject" in result
        assert "From: sender@example.com" in result
        assert "CC:" not in result
        assert "Attachments:" not in result

    @pytest.mark.asyncio
    async def test_create_embedding(self, sample_email, mock_openai_response):
        """Test creating an embedding from email text"""
        embedding_service = EmbeddingService(model_name="test-model")

        # Mock the OpenAI embeddings.create method as an async method
        async def mock_create_async(**kwargs):
            return mock_openai_response

        # Mock the OpenAI embeddings.create method
        with mock.patch("openai.embeddings.create", mock_create_async):
            # Prepare the email text
            email_text = embedding_service.prepare_email_text(sample_email)

            # Call the method
            embedding = await embedding_service.create_embedding(email_text)

            # Verify the result
            assert isinstance(embedding, list)
            assert len(embedding) == 50
            assert embedding[0] == 0.1
            assert embedding[1] == 0.2

    @pytest.mark.asyncio
    async def test_create_embedding_with_long_text(self, mock_openai_response):
        """Test that long text is properly truncated for embedding"""
        embedding_service = EmbeddingService(model_name="test-model")

        # Create a very long text (longer than the max_length in the method)
        long_text = "x" * 10000  # 10,000 characters

        # Create an async mock function
        async def mock_create_async(**kwargs):
            # Capture the input to verify truncation
            mock_create_async.called_with = kwargs
            return mock_openai_response

        # Mock the OpenAI embeddings.create method
        with mock.patch("openai.embeddings.create", mock_create_async):
            # Call the method
            await embedding_service.create_embedding(long_text)

            # Verify the API was called with truncated text
            assert (
                len(mock_create_async.called_with["input"]) == 8000
            )  # Max length in the method

    @pytest.mark.asyncio
    async def test_vectorize_email(self, sample_email, mock_openai_response):
        """Test vectorizing a complete email"""
        embedding_service = EmbeddingService(model_name="test-model")

        # Create an async mock function
        async def mock_create_async(**kwargs):
            return mock_openai_response

        # Mock the OpenAI embeddings.create method
        with mock.patch("openai.embeddings.create", mock_create_async):
            # Vectorize the email
            vectorized_email = await embedding_service.vectorize_email(
                sample_email
            )

            # Verify the result
            assert vectorized_email.email == sample_email
            assert len(vectorized_email.vector) == 50
            assert vectorized_email.embedding_model == "test-model"
            assert isinstance(vectorized_email.vector_id, str)
            assert vectorized_email.date_embedded is not None
