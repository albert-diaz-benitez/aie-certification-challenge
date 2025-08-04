from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EmailAttachment:
    """Represents an email attachment"""

    filename: str
    content_type: str
    content: bytes
    size: int


@dataclass
class EmailMetadata:
    """Represents metadata about an email"""

    email_id: str
    subject: str
    sender: str
    recipients: List[str]
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    date_received: Optional[datetime] = None
    date_processed: Optional[datetime] = None
    has_attachments: bool = False
    importance: str = "normal"
    category: Optional[str] = None
    conversation_id: Optional[str] = None
    size: Optional[int] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Email:
    """Represents an email with content and metadata"""

    metadata: EmailMetadata
    body_text: str
    body_html: Optional[str] = None
    attachments: List[EmailAttachment] = field(default_factory=list)
    embedding_id: Optional[str] = None


@dataclass
class VectorizedEmail:
    """Represents an email that has been vectorized"""

    email: Email
    vector: List[float]
    vector_id: str
    collection_name: str
    embedding_model: str
    date_embedded: datetime = field(default_factory=datetime.now)
