import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Mail settings
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_ACCESS_TOKEN = os.getenv("GMAIL_ACCESS_TOKEN")
GMAIL_TOKEN_EXPIRY = os.getenv("GMAIL_TOKEN_EXPIRY")
GMAIL_USER_EMAIL = os.getenv("GMAIL_USER_EMAIL")
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE")
GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE")

# Email filtering settings
TARGET_SENDERS = os.getenv("TARGET_SENDERS", "").split(",")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "emails")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

# App settings
CRAWL_SCHEDULE = os.getenv(
    "CRAWL_SCHEDULE", "00:00"
)  # Daily at midnight by default
MAX_EMAILS_PER_RUN = int(os.getenv("MAX_EMAILS_PER_RUN", "100"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
