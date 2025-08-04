import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

import qdrant_client
from qdrant_client.http import models as qmodels

from src.config import settings
from src.models.email_models import VectorizedEmail

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(self):
        self.client = qdrant_client.QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure that the collection exists, creating it if necessary"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")

                # Define schema for payload fields we want to filter on
                payload_schema = {
                    "email_id": qmodels.PayloadSchemaType.KEYWORD,
                    "subject": qmodels.PayloadSchemaType.TEXT,
                    "sender": qmodels.PayloadSchemaType.KEYWORD,
                    "date_received": qmodels.PayloadSchemaType.DATETIME,
                    "date_processed": qmodels.PayloadSchemaType.DATETIME,
                    "has_attachments": qmodels.PayloadSchemaType.BOOL,
                    "importance": qmodels.PayloadSchemaType.KEYWORD,
                }

                # Create the collection with proper schema - using correct parameter for current client version
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size, distance=qmodels.Distance.COSINE
                    ),
                )

                # Update the collection with payload schema if supported
                try:
                    # Try the newer API approach
                    for field_name, field_type in payload_schema.items():
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
                    logger.info(
                        f"Collection {self.collection_name} payload schema updated successfully"
                    )
                except Exception as schema_error:
                    logger.warning(
                        f"Failed to create payload schema: {str(schema_error)}"
                    )

                logger.info(
                    f"Collection {self.collection_name} created successfully"
                )
            else:
                logger.info(
                    f"Collection {self.collection_name} already exists"
                )

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def store_email_vector(self, vectorized_email: VectorizedEmail) -> str:
        """
        Store a vectorized email in the database

        Args:
            vectorized_email: The vectorized email to store

        Returns:
            The ID of the stored vector
        """
        try:
            email = vectorized_email.email
            metadata = email.metadata

            # Convert datetime to string for Qdrant
            date_received_str = (
                metadata.date_received.isoformat()
                if metadata.date_received
                else None
            )
            date_processed_str = (
                metadata.date_processed.isoformat()
                if metadata.date_processed
                else None
            )

            # Create payload with metadata for filtering
            payload = {
                "email_id": metadata.email_id,
                "subject": metadata.subject,
                "sender": metadata.sender,
                "recipients": metadata.recipients,
                "date_received": date_received_str,
                "date_processed": date_processed_str,
                "has_attachments": metadata.has_attachments,
                "importance": metadata.importance,
                # Store additional data needed for retrieval
                "body_text": email.body_text[
                    :1000
                ],  # Truncate for payload size
                "body_html_available": email.body_html is not None,
                "attachment_count": len(email.attachments),
                "embedding_model": vectorized_email.embedding_model,
                "embedding_date": vectorized_email.date_embedded.isoformat(),
            }

            # Store the vector in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=vectorized_email.vector_id,
                        vector=vectorized_email.vector,
                        payload=payload,
                    )
                ],
            )

            logger.info(
                f"Stored vector with ID {vectorized_email.vector_id} in collection {self.collection_name}"
            )
            return vectorized_email.vector_id

        except Exception as e:
            logger.error(f"Error storing vector: {str(e)}")
            raise

    def query_by_date_range(
        self,
        start_date: Union[date, datetime],
        end_date: Optional[Union[date, datetime]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query emails by date range

        Args:
            start_date: The start date for the range
            end_date: The end date for the range (defaults to now)
            limit: Maximum number of results to return

        Returns:
            List of matching email records
        """
        try:
            # If end_date not provided, use current date/time
            if end_date is None:
                end_date = datetime.now()

            # Convert date to datetime if needed
            if isinstance(start_date, date) and not isinstance(
                start_date, datetime
            ):
                start_date = datetime.combine(start_date, datetime.min.time())

            if isinstance(end_date, date) and not isinstance(
                end_date, datetime
            ):
                end_date = datetime.combine(end_date, datetime.max.time())

            # Format dates as ISO strings for Qdrant
            start_date_str = start_date.isoformat()
            end_date_str = end_date.isoformat()

            # Build filter for date range using two separate conditions
            # One for gte (greater than or equal to start_date)
            # One for lte (less than or equal to end_date)
            date_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="date_received",
                        range=qmodels.DatetimeRange(
                            gte=start_date_str,
                            gt=None,
                            lt=None,
                            lte=end_date_str,
                        ),
                    )
                ]
            )

            # Query Qdrant
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=date_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,  # We don't need vectors for listing
            )

            # Extract and format results
            emails = []
            for point in results[0]:
                emails.append(point.payload)

            return emails

        except Exception as e:
            logger.error(f"Error querying by date range: {str(e)}")
            raise
