# Task 1: Defining your Problem and Audience

## Context

This solution is designed to work with our current product, Kitsune insurance platform, of inari.io.

The Kitsune platform is an underwriter workbench that helps insurance professionals manage their policies, claims, and client interactions efficiently. The platform integrates various tools and features to streamline the workflow of insurance underwriters, brokers, and agents.

## Problem Statement

Insurance Underwriters need to efficiently extract and process policy information from numerous daily emails to maintain accurate records and provide timely service to clients.

## User Profile and Problem Analysis

**User:** Underwriter

Insurance brokers receive a lot of emails daily containing critical policy information that must be manually extracted, documented, and entered into systems of record. This process is highly time-consuming, error-prone, and diverts valuable attention from client-facing activities. The manual extraction of policy details (insured names, line of business, effective dates, inception dates, and premium amounts) from unstructured email content creates a productivity bottleneck, leading to potential delays in client service and increasing the risk of missing important information.

The current workflow requires underwriters to open each email, read through often lengthy and inconsistently formatted content, identify the key policy parameters, and manually record this information in their systems. This administrative burden reduces the time available for relationship management and advising clients on complex insurance matters - the core value-adding activities of their role. By automating the extraction of policy details from emails, we enable insurance professionals to focus on strategic client interactions while maintaining accurate and up-to-date policy records.

# Task 2: Propose a Solution

## Solution Overview

Our solution is an automated policy information extraction system that integrates seamlessly with the Kitsune insurance platform. The system will automatically monitor underwriters' email inboxes, identify policy-related messages, extract critical information using advanced AI, and populate the Kitsune platform with structured policy data without manual intervention. Underwriters will begin their day with a dashboard showing newly processed policies, organized by priority (based on effective dates and premium values), with all extracted fields pre-populated and ready for final validation. This system transforms the underwriter's workflow from a tedious data entry exercise into a quick verification process, reducing policy information processing time from 15-20 minutes per email to less than 1 minute per policy for final review.

When an email arrives, the underwriter receives a notification showing the extracted policy details and confidence scores. They can quickly approve accurate extractions with a single click or make minimal adjustments where needed. The system learns from these corrections, continuously improving its extraction accuracy. By eliminating manual data entry and reducing processing errors, underwriters reclaim 15-20 hours weekly for value-added client work, improve response times, and increase the consistency and reliability of their policy records within the Kitsune platform.

## Technology Stack

1. **LLM**: GPT-4 through OpenAI's API, chosen for its superior performance in extracting structured information from unstructured text and understanding complex insurance terminology and policy formats.

2. **Embedding Model**: OpenAI's text-embedding-3-small model, selected for its optimal balance of accuracy and cost-efficiency, providing high-quality vector representations of insurance-specific text.

3. **Orchestration**: LangGraph for orchestrating the RAG workflow, complemented by FastAPI for API endpoints, chosen for its built-in state management and intuitive graph-based workflow design that simplifies complex multi-step AI processes.

4. **Vector Database**: Qdrant, selected for its efficient similarity search capabilities, support for metadata filtering (crucial for date-range queries on policies), and excellent performance with Docker containerization.

5. **Monitoring**: LangSmith for tracking LLM calls and performance metrics, providing visibility into the extraction process, confidence scores, and helping identify patterns in extraction failures.

6. **Evaluation**: RAGAS (Retrieval Augmented Generation Assessment) framework for evaluating the RAG pipeline's quality, measuring context relevance, faithfulness, answer relevance, and completeness of the policy information extraction. RAGAS provides standardized metrics that will help us quantify extraction accuracy for each policy field and identify specific areas for improvement.

7. **User Interface**: Integration with the existing Kitsune platform UI. Basically policies will appear on daily basis with a flag to review in case of coming from the emails.

8. **Serving & Inference**: Containerized microservices architecture using Docker Compose for local development and testing, designed for easy deployment to cloud environments with Kubernetes for production scaling.

## Agentic Reasoning

The system employs agentic reasoning in the policy extraction phase, where it must make intelligent decisions about how to interpret ambiguous email content. The agent analyzes the email context to distinguish between different types of policies, determines which sections contain relevant information, and handles variations in how policy details are presented. For example, when encountering a complex email thread with multiple policies mentioned, the agent uses reasoning to:

1. Identify which policy is currently active/relevant in the conversation
2. Determine if numbers represent premiums, limits, or deductibles based on context
3. Handle conflicting information by prioritizing the most recent or authoritative statement

This agentic approach allows the system to handle the wide variety of email formats, writing styles, and complex policy discussions that would be impossible to address with rigid extraction rules or simple prompt engineering.

# Task 3: Dealing with the Data

## Data Sources and External APIs

1. **Email Data Source**: The primary data source is the Gmail API integration that provides access to underwriters' email accounts. This gives us access to the raw, unstructured email content that contains policy information embedded within various formats (text, HTML, occasionally attachments). The Gmail API allows us to filter by date ranges and senders, crucial for targeting relevant emails from insurance carriers and brokers.

2. **Insurance Industry Knowledge Base**: We'll build and maintain a proprietary vector database of insurance industry terminology, standard policy structures, and carrier-specific formatting patterns. This knowledge base will be used to enhance the extraction process by providing context that helps the LLM understand domain-specific terms and identify where critical information typically appears in emails from different carriers.

3. **Kitsune API**: Our internal Kitsune platform API will be used to store the extracted policy information and integrate it with existing client records. The system will check for existing policies with similar attributes to avoid duplication and ensure proper linkage between new and existing policies for the same clients.

## Chunking Strategy

For our email processing system, we'll implement a hybrid chunking strategy that combines structural and semantic approaches:

1. **Email-Level Structure Chunking**: Each email will first be divided into structural components: headers (subject, from, to, date), body text, and attachments. This preserves the natural organization of email data and maintains important context about the source of information.

2. **Content-Aware Semantic Chunking**: Within the body text, we'll implement semantic chunking with a target size of 1,000 tokens and 200 token overlaps. This approach:
   - Uses natural paragraph breaks and section headers when present
   - Keeps intact sections that discuss the same policy details (e.g., maintains all premium information together)
   - Preserves tabular data structures within the same chunk
   - Ensures that date information stays connected to the policy details it relates to

We've selected this chunking strategy because insurance email communication often follows semi-structured formats where related information appears in proximity. The semantic approach prevents splitting critical policy information across multiple chunks, which would make extraction more difficult. The 200-token overlap ensures that context isn't lost between chunks, particularly important for complex policies where details may span multiple paragraphs.

For emails with attachments containing policy information (e.g., PDFs), we'll extract the text and apply document-specific chunking that respects structural elements like headers, sections, and tables commonly found in insurance documentation.

## Additional Data Requirements

1. **Historical Extraction Data**: To train and improve our system over time, we'll maintain a dataset of past extractions with their corrections made by underwriters. This will serve as training data to fine-tune our extraction models and improve accuracy for specific carriers and policy types.

2. **Carrier-Specific Templates**: We'll build a database of templates mapping common formats used by specific insurance carriers to help the system identify where relevant information typically appears. This will be particularly valuable for carriers that use consistent formatting in their communications.

3. **Policy Verification Rules**: We'll create a rules engine containing validation constraints for different policy types (e.g., typical premium ranges for different lines of business, valid effective date ranges). This will help flag potential extraction errors when values fall outside expected parameters.

4. **Confidence Scoring Data**: We'll collect metadata about which parts of emails lead to high-confidence vs. low-confidence extractions. This data will help improve the system's ability to assess its own accuracy and determine when human review is necessary.

# Task 4: Building a Quick End-to-End Agentic RAG Prototype

Implicit on the code

# Task 5: Creating a Golden Test Data Set

TBD

# Task 6: The Benefits of Advanced Retrieval

TBD

# Task 7: Assessing Performance

TBD
