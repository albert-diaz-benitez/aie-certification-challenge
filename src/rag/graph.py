import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

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
    existing_email_data: List[
        Email
    ]  # New field to hold existing emails if any


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
        # Get today's date and start of day
        today = datetime.now()
        start_of_day = datetime.combine(today.date(), datetime.min.time())
        end_of_day = datetime.combine(today.date(), datetime.max.time())

        logger.info(
            f"Checking if emails for {today.date()} are already processed"
        )

        # Check if we already have vectors for today's date in Qdrant
        try:
            existing_emails = qdrant_service.query_by_date_range(
                start_date=start_of_day, end_date=end_of_day
            )

            if existing_emails and len(existing_emails) > 0:
                logger.info(
                    f"Found {len(existing_emails)} already processed emails for today. Skipping retrieval."
                )

                # Instead of skipping directly to no_emails_to_embed, we'll set a status that indicates
                # we should extract policies from the existing data
                return {
                    **state,
                    "emails": [],  # No need to reprocess emails
                    "processed_emails_count": len(existing_emails),
                    "retrieved_date": today.isoformat(),
                    "status": "use_existing_data",  # New status to indicate we have existing data
                    "existing_email_data": existing_emails,  # Pass the existing data along
                }
        except Exception as e:
            # If checking fails, proceed with normal email retrieval
            logger.warning(f"Error checking for existing emails: {str(e)}")

        logger.info("Retrieving today's emails")

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
            "status": "emails_retrieved" if emails else "no_emails_to_embed",
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
    # Check if we should use existing emails data or current emails
    emails_to_process = []
    if (
        "existing_email_data" in state
        and state.get("status") == "use_existing_data"
    ):
        logger.info(
            f"Using {len(state['existing_email_data'])} existing email data for policy extraction"
        )
        emails_to_process = state["existing_email_data"]
    else:
        emails_to_process = state["emails"]

    if not emails_to_process:
        logger.warning("No emails to extract policies from")
        return {**state, "policies": [], "status": "no_emails_for_extraction"}

    try:
        logger.info(
            f"Extracting policy data from {len(emails_to_process)} emails"
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

            Return the data as a valid JSON object with the above fields. Always wrap your response in ```json``` code blocks to ensure proper formatting.""",
                ),
                (
                    "user",
                    "Please extract policy information from this email:\n\nSubject: {subject}\n\nFrom: {sender}\n\nDate: {date}\n\nContent:\n{body_text}",
                ),
            ]
        )

        policies = []

        for email in emails_to_process:
            # Format email for extraction
            email_data = {
                "subject": (
                    email.metadata.subject
                    if hasattr(email, "metadata")
                    else email.get("subject", "Unknown Subject")
                ),
                "sender": (
                    email.metadata.sender
                    if hasattr(email, "metadata")
                    else email.get("sender", "Unknown Sender")
                ),
                "date": (
                    email.metadata.date_received.isoformat()
                    if hasattr(email, "metadata")
                    and email.metadata.date_received
                    else email.get("date", "unknown")
                ),
                "body_text": (
                    email.body_text
                    if hasattr(email, "body_text")
                    else email.get("content", "")
                ),
            }

            try:
                # Generate policy extraction using LLM
                chain = prompt | llm
                result = await chain.ainvoke(email_data)

                # Get the content from the message
                policy_data = result.content

                # Clean up the response to extract valid JSON
                import json
                import re

                # First try to extract JSON from code blocks
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", policy_data, re.DOTALL
                )

                if json_match:
                    policy_data = json_match.group(1)
                else:
                    # If no code blocks, try to find anything that looks like JSON
                    json_match = re.search(r"\{.*\}", policy_data, re.DOTALL)
                    if json_match:
                        policy_data = json_match.group(0)
                    else:
                        # Fallback to using the entire response
                        logger.warning(
                            f"Could not extract JSON from LLM response for email '{email_data['subject']}'"
                        )

                # Try to handle trailing commas in JSON (common LLM mistake)
                policy_data = re.sub(r",\s*}", "}", policy_data)
                policy_data = re.sub(r",\s*]", "]", policy_data)

                # Remove any non-ASCII characters
                policy_data = "".join(c for c in policy_data if ord(c) < 128)

                # Parse the JSON
                try:
                    parsed_policy = json.loads(policy_data)
                except json.JSONDecodeError:
                    # If still can't parse, make one more attempt with a more aggressive approach
                    logger.warning(
                        f"JSON decode error in first attempt for email '{email_data['subject']}'. Attempting more aggressive parsing."
                    )

                    # Create a minimal valid JSON structure
                    parsed_policy = {
                        "policy_insured": None,
                        "line_of_business": None,
                        "effective_date": None,
                        "expected_inception_date": None,
                        "target_premium": None,
                    }

                    # Try to extract individual fields using regex
                    for field in parsed_policy.keys():
                        # Fixed regex pattern by escaping the closing curly brace or using r-string with double braces
                        field_match = re.search(
                            r'"' + field + r'":\s*"?([^",\}]+)"?', policy_data
                        )
                        if field_match:
                            parsed_policy[field] = field_match.group(1).strip()

                # Add source email ID and extraction date
                parsed_policy["source_email_id"] = email.get("sender", "unknown")
                parsed_policy["extraction_date"] = datetime.now().isoformat()

                # Ensure all fields exist
                for field in [
                    "policy_insured",
                    "line_of_business",
                    "effective_date",
                    "expected_inception_date",
                    "target_premium",
                ]:
                    if field not in parsed_policy:
                        parsed_policy[field] = None

                policies.append(parsed_policy)
                logger.debug(
                    f"Successfully extracted policy data from email: {email_data['subject']}"
                )

            except Exception as extraction_error:
                logger.warning(
                    f"Failed to extract policy from email '{email_data['subject']}': {str(extraction_error)}"
                )
                # Add a minimal entry with error information
                policies.append(
                    {
                        "policy_insured": None,
                        "line_of_business": None,
                        "effective_date": None,
                        "expected_inception_date": None,
                        "target_premium": None,
                        "source_email_id": (
                            email.metadata.email_id
                            if hasattr(email, "metadata")
                            else email.get("id", "unknown")
                        ),
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
    elif state["status"] == "use_existing_data":
        # Skip directly to extract_policies when we have existing data
        return "extract_policies"
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

    # Add the START edge to the first step (retrieve_emails)
    workflow.add_edge(START, "retrieve_emails")

    # Add conditional edges
    workflow.add_conditional_edges(
        "retrieve_emails",
        should_end,
        {
            "store_embeddings": "store_embeddings",
            "extract_policies": "extract_policies",
            "end": END,
        },
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
        existing_email_data=[]
    )

    # Create a memory saver for checkpoints
    memory = InMemorySaver()

    # Execute the graph - using ainvoke instead of acall for newer LangGraph versions
    result = await graph.ainvoke(
        initial_state, config={"checkpointer": memory}
    )

    return result


# For testing and evaluation
async def extract_policy_field(email_text: str, field: str) -> str:
    """
    Extract a specific policy field from an email for evaluation purposes.

    Args:
        email_text: The raw email text
        field: The specific field to extract (e.g., "policy_insured")

    Returns:
        The extracted value for the requested field
    """
    # Initialize the language model
    eval_llm = ChatOpenAI(model="gpt-4")

    # Create a simple prompt focused on extracting just the requested field
    field_descriptions = {
        "policy_insured": "the name of the person, business, or entity that is insured by the policy",
        "line_of_business": "the type of insurance or line of business (e.g., Property, Auto, Liability)",
        "effective_date": "the date when the policy becomes effective or active, in YYYY-MM-DD format if possible",
        "expected_inception_date": "the date when the policy is expected to begin or be implemented, in YYYY-MM-DD format if possible",
        "target_premium": "the premium amount for the policy, including any currency symbols",
    }

    description = field_descriptions.get(field, f"the {field}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert insurance data extractor. Your task is to extract {description} from the provided email.

        Extract ONLY this specific piece of information. If the information is not present in the email, respond with "Not found".

        Return ONLY the extracted value, with no additional text or explanation.""",
            ),
            (
                "user",
                f"Please extract {description} from this email:\n\n{email_text}",
            ),
        ]
    )

    # Extract the field using ainvoke (async version) instead of invoke
    chain = prompt | eval_llm
    result = await chain.ainvoke(
        {}
    )  # Use empty dict since no input variables in the prompt template
    time.sleep(2)  # avoid max requests error in OpenAI API

    # Return just the content
    return (
        result.content.strip()
        if hasattr(result, "content")
        else str(result).strip()
    )


# For testing
if __name__ == "__main__":
    asyncio.run(run_policy_extraction_graph())
