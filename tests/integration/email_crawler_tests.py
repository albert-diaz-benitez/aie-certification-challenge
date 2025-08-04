import unittest
from datetime import datetime, timedelta

from src.services.email_crawler import EmailCrawler


class TestEmailCrawler(unittest.TestCase):
    """Integration tests for the GmailCrawler service"""

    def setUp(self):
        self.crawler = EmailCrawler()

    def test_authenticate_with_credentials_file(self):
        result = self.crawler.authenticate()

        assert result == True

    def test_get_emails_basic(self):
        """Test basic email retrieval"""
        since_date = datetime.now() - timedelta(days=1)
        emails = self.crawler.get_emails(since_date=since_date, max_emails=10)

        assert len(emails) > 0

        for email in emails:
            if (
                email.metadata.sender == "albert.diaz@inari.io"
                and email.metadata.subject == "Test email"
            ):
                assert True
                return

        assert False

    def test_get_emails_with_sender_filter(self):
        since_date = datetime.now() - timedelta(days=1)
        emails = self.crawler.get_emails(
            since_date=since_date,
            max_emails=10,
            target_senders=["albert.diaz@inari.io"],
        )

        assert len(emails) > 0

        for email in emails:
            if (
                email.metadata.sender == "albert.diaz@inari.io"
                and email.metadata.subject == "Test email"
            ):
                assert True
                return

        assert False
