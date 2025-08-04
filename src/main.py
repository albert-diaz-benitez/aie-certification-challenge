import logging
from datetime import datetime, timedelta

from src.config import settings
from src.services.email_crawler import EmailCrawler
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmailProcessingService:
    """Main service that orchestrates email crawling, embedding, and storage"""

    def __init__(self):
        self.email_crawler = EmailCrawler()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    async def process_emails(self):
        """Process emails: fetch, embed, and store them"""
        try:
            logger.info("Starting email processing job")

            # Get target senders from config
            target_senders = (
                settings.TARGET_SENDERS if settings.TARGET_SENDERS else []
            )
            if not target_senders:
                logger.warning(
                    "No target senders configured, will process all emails"
                )

            # Get emails from last 24 hours by default
            since_date = datetime.now() - timedelta(days=1)

            # Fetch emails from Gmail
            logger.info(f"Fetching emails since {since_date}")
            emails = self.email_crawler.get_emails(
                since_date=since_date,
                max_emails=settings.MAX_EMAILS_PER_RUN,
                target_senders=target_senders,
            )

            if not emails:
                logger.info("No emails found matching the criteria")
                return

            logger.info(f"Processing {len(emails)} emails")

            # Process each email
            for email in emails:
                try:
                    # Create vector embedding
                    logger.debug(
                        f"Creating embedding for email: {email.metadata.subject}"
                    )
                    vectorized_email = (
                        await self.embedding_service.vectorize_email(email)
                    )

                    # Store in Qdrant
                    logger.debug("Storing vector in Qdrant")
                    vector_id = self.qdrant_service.store_email_vector(
                        vectorized_email
                    )

                    logger.info(
                        f"Successfully processed email: {email.metadata.subject} (Vector ID: {vector_id})"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing email {email.metadata.subject}: {str(e)}"
                    )
                    continue

            logger.info("Email processing job completed")

        except Exception as e:
            logger.error(f"Error in email processing job: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Email Processing Service")
    service = EmailProcessingService()
