import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.config.settings import TARGET_SENDERS
from src.models.email_models import Email
from src.services.email_crawler import EmailCrawler
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define our graph state
class PolicyExtractionState(TypedDict):
    emails: List[Email]
    processed_emails_count: int
    retrieved_date: str
    policies: List[Dict[str, Any]]
    errors: List[str]
    status: str


# Initialize services
email_crawler = EmailCrawler()
embedding_service = EmbeddingService()
qdrant_service = QdrantService()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", temperature=0.1)


# Define the steps in our graph
async def retrieve_emails(
    state: PolicyExtractionState,
) -> PolicyExtractionState:
    """
    Retrieve today's emails from Gmail
    """
    try:
        logger.info("Retrieving today's emails")

        # Get today's date and start of day
        today = datetime.now()
        start_of_day = datetime.combine(today.date(), datetime.min.time())

        # Get target senders from environment
        target_senders = TARGET_SENDERS
        if not target_senders or target_senders == [""]:
            logger.warning(
                "No target senders specified, retrieving all emails"
            )

        # Retrieve emails from today
        emails = email_crawler.get_emails(
            since_date=start_of_day,
            target_senders=target_senders if target_senders != [""] else None,
        )

        logger.info(f"Retrieved {len(emails)} emails from today")

        return {
            **state,
            "emails": emails,
            "processed_emails_count": len(emails),
            "retrieved_date": today.isoformat(),
            "status": "emails_retrieved",
        }
    except Exception as e:
        logger.error(f"Error retrieving emails: {str(e)}")
        return {
            **state,
            "emails": [],
            "errors": [*state["errors"], f"Email retrieval error: {str(e)}"],
            "status": "email_retrieval_failed",
        }


async def store_embeddings(
    state: PolicyExtractionState,
) -> PolicyExtractionState:
    """
    Create embeddings for the retrieved emails and store them in Qdrant
    """
    if not state["emails"]:
        logger.warning("No emails to embed")
        return {**state, "status": "no_emails_to_embed"}

    try:
        logger.info(f"Creating embeddings for {len(state['emails'])} emails")

        vector_ids = []
        for email in state["emails"]:
            # Create embedding for email
            vectorized_email = await embedding_service.vectorize_email(email)

            # Store in Qdrant
            vector_id = qdrant_service.store_email_vector(vectorized_email)
            vector_ids.append(vector_id)

        logger.info(
            f"Successfully stored {len(vector_ids)} email embeddings in Qdrant"
        )

        return {**state, "status": "embeddings_stored"}
    except Exception as e:
        logger.error(f"Error creating or storing embeddings: {str(e)}")
        return {
            **state,
            "errors": [*state["errors"], f"Embedding error: {str(e)}"],
            "status": "embedding_failed",
        }


async def extract_policy_data(
    state: PolicyExtractionState,
) -> PolicyExtractionState:
    """
    Extract policy information from the emails using an LLM
    """
    if not state["emails"]:
        logger.warning("No emails to extract policies from")
        return {**state, "policies": [], "status": "no_emails_for_extraction"}

    try:
        logger.info(
            f"Extracting policy data from {len(state['emails'])} emails"
        )

        # Template for policy extraction
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert insurance data extractor. Your task is to analyze email content and extract structured policy information. You must only use the provided email data, and cannot use your own knowledge

            Extract the following fields if present:
            - policy_insured (the name of the person or entity insured)
            - line_of_business (e.g., Auto, Home, Life, Commercial, Property, etc.)
            - effective_date (when the policy becomes effective)
            - expected_inception_date (when the policy is expected to begin)
            - target_premium (the premium amount)

            Format dates as YYYY-MM-DD if possible.
            If a field is not found in the email, return null for that field.

            Return the data as a valid JSON object with the above fields.""",
                ),
                (
                    "user",
                    "Please extract policy information from this email:\n\nSubject: {subject}\n\nFrom: {sender}\n\nDate: {date}\n\nContent:\n{body_text}",
                ),
            ]
        )

        policies = []

        for email in state["emails"]:
            # Format email for extraction
            email_data = {
                "subject": email.metadata.subject,
                "sender": email.metadata.sender,
                "date": (
                    email.metadata.date_received.isoformat()
                    if email.metadata.date_received
                    else "unknown"
                ),
                "body_text": email.body_text,
            }

            # Generate policy extraction using LLM
            chain = prompt | llm
            result = await chain.invoke(email_data)

            try:
                # Parse the policy data
                policy_data = result.content

                # Clean up the response if needed (assuming it's JSON or can be parsed as such)
                import json
                import re

                # Try to extract JSON from the response if it's not pure JSON
                json_match = re.search(
                    r"```json\n(.*?)\n```", policy_data, re.DOTALL
                )
                if json_match:
                    policy_data = json_match.group(1)

                # Parse the JSON
                parsed_policy = json.loads(policy_data)

                # Add source email ID and extraction date
                parsed_policy["source_email_id"] = email.metadata.email_id
                parsed_policy["extraction_date"] = datetime.now().isoformat()

                policies.append(parsed_policy)
                logger.debug(
                    f"Successfully extracted policy data from email: {email.metadata.subject}"
                )

            except Exception as extraction_error:
                logger.warning(
                    f"Failed to extract policy from email '{email.metadata.subject}': {str(extraction_error)}"
                )
                # Add a minimal entry with error information
                policies.append(
                    {
                        "policy_insured": None,
                        "line_of_business": None,
                        "effective_date": None,
                        "expected_inception_date": None,
                        "target_premium": None,
                        "source_email_id": email.metadata.email_id,
                        "extraction_date": datetime.now().isoformat(),
                        "confidence_score": 0.0,
                        "extraction_error": str(extraction_error),
                    }
                )

        logger.info(f"Extracted information for {len(policies)} policies")

        return {**state, "policies": policies, "status": "policies_extracted"}

    except Exception as e:
        logger.error(f"Error extracting policy data: {str(e)}")
        return {
            **state,
            "errors": [*state["errors"], f"Policy extraction error: {str(e)}"],
            "status": "extraction_failed",
            "policies": [],  # Ensure we have an empty policies list
        }


def should_end(state: PolicyExtractionState) -> str:
    """
    Determine if the workflow should end or continue to the next step
    """
    # End conditions
    if state["status"] in [
        "policies_extracted",
        "extraction_failed",
        "no_emails_for_extraction",
    ]:
        return "end"

    # Check if email retrieval failed
    if state["status"] == "email_retrieval_failed":
        return "end"

    # Check if embedding failed
    if state["status"] == "embedding_failed":
        return "end"

    # Default flow
    if state["status"] == "emails_retrieved":
        return "store_embeddings"
    elif (
        state["status"] == "embeddings_stored"
        or state["status"] == "no_emails_to_embed"
    ):
        return "extract_policies"

    # If unsure, just end
    return "end"


# Build the LangGraph workflow
def build_policy_extraction_graph():
    """
    Create the LangGraph workflow for policy extraction
    """
    # Initialize the graph with our state
    workflow = StateGraph(PolicyExtractionState)

    # Add nodes for each step
    workflow.add_node("retrieve_emails", retrieve_emails)
    workflow.add_node("store_embeddings", store_embeddings)
    workflow.add_node("extract_policies", extract_policy_data)

    # Add conditional edges
    workflow.add_conditional_edges(
        "retrieve_emails",
        should_end,
        {"store_embeddings": "store_embeddings", "end": END},
    )

    workflow.add_conditional_edges(
        "store_embeddings",
        should_end,
        {"extract_policies": "extract_policies", "end": END},
    )

    # Add final edge
    workflow.add_edge("extract_policies", END)

    # Compile the graph
    return workflow.compile()


# Function to run the graph
async def run_policy_extraction_graph():
    """
    Run the policy extraction graph and return the results
    """
    # Create the graph
    graph = build_policy_extraction_graph()

    # Initialize state
    initial_state = PolicyExtractionState(
        emails=[],
        processed_emails_count=0,
        retrieved_date="",
        policies=[],
        errors=[],
        status="initialized",
    )

    # Create a memory saver for checkpoints
    memory = InMemorySaver()

    # Execute the graph
    result = await graph.acall(initial_state, config={"checkpointer": memory})

    return result


# For testing
if __name__ == "__main__":
    asyncio.run(run_policy_extraction_graph())
