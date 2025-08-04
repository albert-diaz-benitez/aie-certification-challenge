import logging
from datetime import datetime
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.graph import run_policy_extraction_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Policy Extraction API",
    description="API for crawling and extracting policy information from emails",
    version="1.0.0",
)


# Define response models
class PolicyData(BaseModel):
    """Model representing extracted policy data"""

    policy_insured: Optional[str] = None
    line_of_business: Optional[str] = None
    effective_date: Optional[str] = None
    expected_inception_date: Optional[str] = None
    target_premium: Optional[str] = None
    source_email_id: Optional[str] = None
    extraction_date: str


class CrawlPoliciesResponse(BaseModel):
    """Response model for crawl-policies-data endpoint"""

    policies: List[PolicyData]
    processed_emails_count: int
    extraction_timestamp: str


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "policy-extraction-api"}


@app.post("/crawl-policies-data", response_model=CrawlPoliciesResponse)
async def crawl_policies_data(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger email crawling and policy extraction

    This will:
    1. Retrieve today's emails
    2. Store them in Qdrant
    3. Extract policy information using LLM
    4. Return the extracted policies as structured data
    """
    try:
        logger.info("Starting policy extraction workflow")

        # Run the LangGraph workflow
        result = await run_policy_extraction_graph()

        # Log completion
        logger.info(
            f"Successfully extracted {len(result['policies'])} policies from {result['processed_emails_count']} emails"
        )

        # Return the response
        return CrawlPoliciesResponse(
            policies=result["policies"],
            processed_emails_count=result["processed_emails_count"],
            extraction_timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in policy extraction workflow: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Policy extraction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
