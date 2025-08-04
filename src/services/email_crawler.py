import base64
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import html2text
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.email_models import Email, EmailAttachment, EmailMetadata

logger = logging.getLogger(__name__)


class EmailCrawler:
    """Service to crawl and extract emails from Gmail"""

    def __init__(self):
        self.creds = None
        self.service = None
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True

    def authenticate(self) -> bool:
        """Authenticate with Gmail API"""
        try:
            # Check if token file exists
            if os.path.exists(settings.GMAIL_TOKEN_FILE):
                with open(settings.GMAIL_TOKEN_FILE) as token:
                    token_data = json.load(token)
                    self.creds = Credentials.from_authorized_user_info(
                        token_data, settings.GMAIL_SCOPES
                    )

            # If no valid credentials available, let's either refresh or create new ones
            if not self.creds or not self.creds.valid:
                if (
                    self.creds
                    and self.creds.expired
                    and self.creds.refresh_token
                ):
                    # If token is expired but we have a refresh token, refresh it
                    self.creds.refresh(Request())
                else:
                    # If we have a credentials file, use it for the flow
                    if os.path.exists(settings.GMAIL_CREDENTIALS_FILE):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            settings.GMAIL_CREDENTIALS_FILE,
                            settings.GMAIL_SCOPES,
                        )
                        self.creds = flow.run_local_server(port=0)
                    else:
                        # If no credentials file, try to create credentials from environment variables
                        if (
                            settings.GMAIL_CLIENT_ID
                            and settings.GMAIL_CLIENT_SECRET
                        ):
                            # Create a credentials file from environment variables
                            client_config = {
                                "installed": {
                                    "client_id": settings.GMAIL_CLIENT_ID,
                                    "client_secret": settings.GMAIL_CLIENT_SECRET,
                                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                    "token_uri": "https://oauth2.googleapis.com/token",
                                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                                    "redirect_uris": [
                                        "urn:ietf:wg:oauth:2.0:oob",
                                        "http://localhost",
                                    ],
                                }
                            }

                            # Save to a file for the flow
                            with open(
                                settings.GMAIL_CREDENTIALS_FILE, "w"
                            ) as f:
                                json.dump(client_config, f)

                            flow = InstalledAppFlow.from_client_secrets_file(
                                settings.GMAIL_CREDENTIALS_FILE,
                                settings.GMAIL_SCOPES,
                            )
                            self.creds = flow.run_local_server(port=0)
                        else:
                            logger.error(
                                "No credentials file and no client ID/secret in environment variables"
                            )
                            return False

                # Save the credentials for the next run
                with open(settings.GMAIL_TOKEN_FILE, "w") as token:
                    token.write(self.creds.to_json())

            # Create the Gmail API service
            self.service = build("gmail", "v1", credentials=self.creds)
            logger.info("Successfully authenticated with Gmail API")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False

    @property
    def gmail_service(self):
        """Get the Gmail service, authenticating if necessary"""
        if not self.service:
            if not self.authenticate():
                raise ValueError("Failed to authenticate with Gmail")
        return self.service

    def _parse_email_date(self, date_str: str) -> datetime:
        """Parse the email date string to datetime object"""
        try:
            # Gmail API returns RFC 2822 formatted date strings
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(date_str)
        except Exception:
            logger.warning(f"Failed to parse date: {date_str}")
            return datetime.now()

    def _get_email_body(self, parts: List) -> tuple:
        """Extract email body (text and html) from message parts"""
        body_html = None
        body_text = None

        def extract_body(parts):
            nonlocal body_html, body_text

            for part in parts:
                if (
                    part.get("mimeType") == "text/plain"
                    and part.get("body")
                    and part.get("body").get("data")
                ):
                    data = base64.urlsafe_b64decode(
                        part.get("body").get("data").encode("ASCII")
                    )
                    body_text = data.decode("utf-8")

                elif (
                    part.get("mimeType") == "text/html"
                    and part.get("body")
                    and part.get("body").get("data")
                ):
                    data = base64.urlsafe_b64decode(
                        part.get("body").get("data").encode("ASCII")
                    )
                    body_html = data.decode("utf-8")

                elif part.get("parts"):
                    extract_body(part.get("parts"))

        if parts:
            extract_body(parts)

        # If we have HTML but no plain text, convert HTML to text
        if body_html and not body_text:
            body_text = self.html_converter.handle(body_html)

        return body_text, body_html

    def _get_attachments(
        self, message_id: str, parts: List
    ) -> List[EmailAttachment]:
        """Extract attachments from email parts"""
        attachments = []

        def extract_attachments(parts):
            for part in parts:
                if (
                    part.get("filename")
                    and part.get("body")
                    and part.get("body").get("attachmentId")
                ):
                    try:
                        attachment_id = part.get("body").get("attachmentId")
                        attachment = (
                            self.gmail_service.users()
                            .messages()
                            .attachments()
                            .get(
                                userId="me",
                                messageId=message_id,
                                id=attachment_id,
                            )
                            .execute()
                        )

                        if attachment.get("data"):
                            data = base64.urlsafe_b64decode(
                                attachment.get("data").encode("ASCII")
                            )

                            email_attachment = EmailAttachment(
                                filename=part.get("filename"),
                                content_type=part.get("mimeType"),
                                content=data,
                                size=len(data),
                            )
                            attachments.append(email_attachment)
                    except Exception as e:
                        logger.warning(
                            f"Failed to download attachment: {str(e)}"
                        )

                if part.get("parts"):
                    extract_attachments(part.get("parts"))

        if parts:
            extract_attachments(parts)

        return attachments

    def _extract_email_address(self, header_value: str) -> List[str]:
        """Extract email addresses from header value"""
        if not header_value:
            return []

        # Simple parsing for email addresses in the format "Name <email@example.com>"
        addresses = []
        parts = header_value.split(",")

        for part in parts:
            part = part.strip()
            # Try to extract email from "Name <email@example.com>" format
            if "<" in part and ">" in part:
                email = part.split("<")[1].split(">")[0]
                addresses.append(email)
            else:
                # If no angle brackets, assume it's just an email
                addresses.append(part)

        return addresses

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_emails(
        self,
        since_date: Optional[datetime] = None,
        max_emails: int = 100,
        target_senders: Optional[List[str]] = None,
    ) -> List[Email]:
        """
        Fetch emails from Gmail with optional filtering

        Args:
            since_date: Only fetch emails received after this date
            max_emails: Maximum number of emails to fetch
            target_senders: If provided, only fetch emails from these senders

        Returns:
            List of Email objects
        """
        try:
            # Default to fetching emails from the last 24 hours if not specified
            if since_date is None:
                since_date = datetime.now() - timedelta(days=1)

            # Build query for Gmail API
            query = f"after:{int(since_date.timestamp())}"

            if target_senders and len(target_senders) > 0:
                sender_filters = " OR ".join(
                    [f"from:{sender}" for sender in target_senders]
                )
                query += f" AND ({sender_filters})"

            # Execute the search
            results = (
                self.gmail_service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_emails)
                .execute()
            )

            messages = results.get("messages", [])
            logger.info(f"Found {len(messages)} messages matching criteria")

            # Process messages into Email objects
            emails = []
            for msg_data in messages:
                # Get full message details
                msg = (
                    self.gmail_service.users()
                    .messages()
                    .get(userId="me", id=msg_data["id"], format="full")
                    .execute()
                )

                # Extract headers
                headers = {
                    header["name"]: header["value"]
                    for header in msg["payload"]["headers"]
                }

                # Parse email metadata
                subject = headers.get("Subject", "[No Subject]")
                sender = headers.get("From", "unknown")
                sender_email = (
                    self._extract_email_address(sender)[0]
                    if self._extract_email_address(sender)
                    else "unknown"
                )

                # Process recipients
                to_recipients = self._extract_email_address(
                    headers.get("To", "")
                )
                cc_recipients = self._extract_email_address(
                    headers.get("Cc", "")
                )
                bcc_recipients = self._extract_email_address(
                    headers.get("Bcc", "")
                )

                # Get date
                date_str = headers.get("Date")
                date_received = (
                    self._parse_email_date(date_str)
                    if date_str
                    else datetime.now()
                )

                # Create metadata
                metadata = EmailMetadata(
                    email_id=msg["id"],
                    subject=subject,
                    sender=sender_email,
                    recipients=to_recipients,
                    cc=cc_recipients,
                    bcc=bcc_recipients,
                    date_received=date_received,
                    date_processed=datetime.now(),
                    has_attachments="filename" in str(msg["payload"]),
                    importance="normal",  # Gmail doesn't have direct importance like Outlook
                    conversation_id=(
                        msg["threadId"] if "threadId" in msg else None
                    ),
                    size=len(str(msg)),  # Approximation of size
                )

                # Get body content and attachments
                parts = [msg["payload"]] if "payload" in msg else []
                body_text, body_html = self._get_email_body(parts)
                attachments = self._get_attachments(msg["id"], parts)

                # Create Email object
                email = Email(
                    metadata=metadata,
                    body_text=body_text or "",
                    body_html=body_html,
                    attachments=attachments,
                )

                emails.append(email)

            return emails

        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            raise
