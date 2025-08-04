import logging
import uuid
from datetime import datetime
from typing import List, Optional

import openai

from src.config import settings
from src.models.email_models import Email, VectorizedEmail

logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = settings.OPENAI_API_KEY


class EmbeddingService:
    """Service to create vector embeddings from email content"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.OPENAI_EMBEDDING_MODEL

    def prepare_email_text(self, email: Email) -> str:
        """
        Prepare email text for embedding by combining relevant fields

        Args:
            email: The email object to prepare

        Returns:
            A string representation of the email suitable for embedding
        """
        parts = [
            f"Subject: {email.metadata.subject}",
            f"From: {email.metadata.sender}",
            f"Date: {email.metadata.date_received.isoformat() if email.metadata.date_received else 'unknown'}",
            f"To: {', '.join(email.metadata.recipients)}",
        ]

        if email.metadata.cc and len(email.metadata.cc) > 0:
            parts.append(f"CC: {', '.join(email.metadata.cc)}")

        # Add the email body
        parts.append("\nContent:")
        parts.append(email.body_text)

        # Add attachment information
        if email.attachments:
            parts.append("\nAttachments:")
            for att in email.attachments:
                parts.append(
                    f"- {att.filename} ({att.content_type}, {att.size} bytes)"
                )

        # Join all parts with newlines
        return "\n".join(parts)

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        try:
            # Truncate text if it's too long (OpenAI has token limits)
            # This is a simple truncation - a more sophisticated approach could
            # summarize or chunk the text while preserving important information
            max_length = 8000  # Approximate limit for embedding models
            if len(text) > max_length:
                logger.warning(
                    f"Text too long ({len(text)} chars), truncating to {max_length}"
                )
                text = text[:max_length]

            response = openai.embeddings.create(
                model=self.model_name, input=text
            )

            # Extract the embedding vector
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    async def vectorize_email(self, email: Email) -> VectorizedEmail:
        """
        Convert an email to a vectorized representation

        Args:
            email: The email to vectorize

        Returns:
            A VectorizedEmail object with the embedding vector
        """
        # Prepare text for embedding
        text = self.prepare_email_text(email)

        # Create embedding
        vector = await self.create_embedding(text)

        # Generate a unique ID for this vector
        vector_id = str(uuid.uuid4())

        # Create vectorized email object
        vectorized_email = VectorizedEmail(
            email=email,
            vector=vector,
            vector_id=vector_id,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding_model=self.model_name,
            date_embedded=datetime.now(),
        )

        return vectorized_email
