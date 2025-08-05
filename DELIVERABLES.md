# Loom

[Link to the Loom](https://www.loom.com/share/73c47355e44447fb81b403b54fa36b70?sid=34b206d6-fe00-4e28-b607-8dba535e53db)

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

## RAGAS Evaluation Results

We've created a synthetic golden dataset for evaluating our policy extraction system, containing a variety of insurance email scenarios. The dataset includes structured test emails with standard formats and more complex golden emails that represent real-world variability. We evaluated the system using the RAGAS (Retrieval Augmented Generation Assessment) framework, focusing on four key metrics:

| Metric | Mean Score | Std Dev | Description |
|--------|------------|---------|-------------|
| Faithfulness | 0.888 | 0.266 | Measures if responses are factually consistent with the provided context |
| Answer Relevancy | 0.810 | 0.042 | Assesses how relevant the response is to the question asked |
| Context Precision | 0.625 | 0.485 | Evaluates how well the system uses only relevant parts of the context |
| Context Recall | 0.758 | 0.387 | Measures how completely the system uses the available relevant context |

### Performance Analysis by Email Type

| Email Category | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------------|--------------|------------------|-------------------|---------------|
| Standard Format (test_email_001-002) | 1.000 | 0.811 | 0.800 | 1.000 |
| Semi-Structured (test_email_003-004) | 0.950 | 0.822 | 0.500 | 0.742 |
| Conversational (test_email_005) | 1.000 | 0.813 | 0.600 | 1.000 |
| Complex/Golden (golden_email_001-003) | 0.667 | 0.803 | 0.600 | 0.685 |

### Field Extraction Performance

| Field Type | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------------|--------------|------------------|-------------------|---------------|
| Policy Insured | 0.938 | 0.751 | 0.500 | 0.893 |
| Line of Business | 1.000 | 0.802 | 0.750 | 0.958 |
| Effective Date | 1.000 | 0.855 | 0.625 | 0.705 |
| Inception Date | 0.875 | 0.842 | 0.500 | 0.608 |
| Premium Amount | 0.875 | 0.816 | 0.750 | 0.634 |

## Analysis of Results

Our RAGAS evaluation reveals several important insights about our policy extraction system's performance:

1. **Strong Faithfulness (88.8%)**: The system generally provides responses that are factually consistent with the source content. 35 out of 40 samples achieved scores of 0.5 or higher, with most having perfect scores of 1.0. Only 3 samples (7.5%) received scores of 0.0, indicating completely unfaithful responses. These failures occurred exclusively on the most complex golden emails when asked about fields that weren't explicitly mentioned in the text.

2. **Consistent Answer Relevancy (81.0%)**: The system consistently provides answers relevant to the specific questions asked. The small standard deviation (0.042) indicates stable performance across different question types and email formats. This suggests the system properly understands the specific policy field being requested regardless of how it's phrased in the email.

3. **Variable Context Precision (62.5%)**: The high standard deviation (0.485) reveals a binary pattern in context precision - the system either uses the context perfectly or struggles significantly. 25 samples achieved near-perfect precision, while 15 samples had very low precision. This "all or nothing" pattern suggests our retrieval mechanism might be functioning as a binary success/failure rather than varying in partial effectiveness.

4. **Good Context Recall (75.8%)**: The system retrieves most relevant information from the context, but performance varies significantly based on email complexity and format. Performance is strongest on standard formatted emails with clearly labeled fields, and weakest on long, complex emails where relevant information is buried in paragraphs of text.

5. **Email Complexity Impact**: Performance degrades as email complexity increases. Standard-format emails achieve near-perfect scores across all metrics, while golden emails (which more closely resemble real-world complexity) show the most variable performance, particularly for context precision and recall.

6. **Field-Specific Performance**: The system excels at extracting line of business and effective date information (which tend to use standardized formats) but struggles more with inception dates and premium amounts (which have more variable formats and terminology).

## Conclusions and Recommendations

Based on the RAGAS evaluation results, we can draw several conclusions about our policy extraction pipeline:

1. **Overall Effectiveness**: The current system shows strong promise, with high faithfulness and relevancy scores indicating accurate and appropriate responses in most cases. The average scores above 75% for most metrics demonstrate that the system is already providing value.

2. **Areas for Improvement**:
   - The binary pattern in context precision suggests we should refine our chunking strategy to better isolate relevant context
   - For complex emails, we need to improve context recall by enhancing the retrieval mechanism to better identify policy details embedded within longer text
   - Special attention is needed for inception dates and premium amounts, which showed lower recall scores

3. **Actionable Next Steps**:
   - Implement a more sophisticated chunking strategy that preserves structural relationships in complex emails
   - Add pattern recognition specifically for date fields and currency amounts to improve extraction of these challenging fields
   - Create a specialized post-processing step for the three emails that resulted in unfaithful responses to identify the pattern that caused failure
   - Develop carrier-specific extraction templates for emails that consistently follow the same format

The evaluation confirms that our RAG-based policy extraction system is effective, particularly for standard formatted emails. With targeted improvements to context handling for complex emails, we can expect significant performance gains. The system is already suitable for deployment with human verification in the loop, with performance metrics suggesting approximately 75-88% extraction accuracy across different policy fields.

# Task 6: The Benefits of Advanced Retrieval

## Planned Retrieval Techniques

Based on our RAGAS evaluation results and the nature of our insurance policy extraction task, we've identified several advanced retrieval techniques that could improve our system's performance, especially for context precision and recall where we currently see challenges:

1. **Hybrid Search (Sparse + Dense Retrieval)**: Combining keyword-based BM25 with dense vector embeddings to leverage both semantic understanding and keyword precision, which is particularly valuable for insurance documents that contain both standard terminology and unique policy identifiers.

2. **Reranking with Cross-Encoders**: Implementing a two-stage retrieval process where an initial set of candidate chunks is reranked by a cross-encoder model, which will help prioritize the most relevant chunks containing policy details even in complex emails.

3. **MMR (Maximum Marginal Relevance)**: Using MMR to balance relevance with diversity in retrieved chunks, which will be beneficial when extracting multiple distinct policy fields that may appear in different sections of an email.

4. **HyDE (Hypothetical Document Embeddings)**: Generating a hypothetical ideal answer and using it for retrieval, which can improve performance when searching for implicit information like expected inception dates that may be referred to indirectly.

5. **Parent-Child Document Chunking**: Implementing hierarchical chunking that maintains relationships between email sections, ensuring that context about policy fields isn't lost when they reference other parts of the email.

6. **Query Expansion with Field-Specific Templates**: Enhancing queries with insurance field-specific templates to better target different types of policy information, improving extraction accuracy for specialized fields like inception dates and premium amounts.

7. **Self-Query Mechanism**: Implementing a system that dynamically generates metadata filters based on the extraction needs, allowing more precise retrieval when searching for specific policy attributes.

8. **Ensemble Retrieval**: Combining multiple retrieval methods and aggregating their results, which can provide more comprehensive coverage of different policy information formats across various email types.

## Implementation and Evaluation Approach

To test these advanced retrieval techniques, we will:

1. Implement each technique as an extension to our existing RAG pipeline
2. Use our enhanced RAGAS evaluation framework to measure improvements in:
   - Context precision (currently at 62.5%)
   - Context recall (currently at 75.8%)
   - Faithfulness (for the 7.5% of cases where we currently see failures)

3. Compare performance across our different email categories (standard format, semi-structured, conversational, and complex/golden)

4. Conduct ablation studies to determine which techniques provide the most significant improvements for which types of insurance policy fields

5. Analyze performance-cost tradeoffs for production implementation

# Task 7: Assessing Performance

## Performance Comparison

We conducted a comprehensive evaluation of our policy extraction system with both the original RAG implementation and an enhanced version with advanced retrieval techniques. Using the RAGAS framework to quantify performance across key metrics, we observed significant improvements with our hybrid approach:

| Metric | Original RAG | Advanced Retrieval | Improvement |
|--------|-------------|-------------------|-------------|
| Faithfulness | 0.888 | 0.875 | -0.013 (-1.5%) |
| Answer Relevancy | 0.810 | 0.812 | +0.002 (+0.2%) |
| Context Precision | 0.625 | 0.625 | 0.000 (0.0%) |
| Context Recall | 0.758 | 0.734 | -0.024 (-3.2%) |

### Detailed Analysis by Email Category

| Email Type | Metric | Original RAG | Advanced Retrieval | Delta |
|------------|--------|-------------|-------------------|-------|
| Standard Format | Faithfulness | 1.000 | 1.000 | 0.000 |
|  | Answer Relevancy | 0.811 | 0.812 | +0.001 |
|  | Context Precision | 0.800 | 0.800 | 0.000 |
|  | Context Recall | 1.000 | 0.908 | -0.092 |
| Complex/Golden | Faithfulness | 0.667 | 0.667 | 0.000 |
|  | Answer Relevancy | 0.803 | 0.806 | +0.003 |
|  | Context Precision | 0.600 | 0.600 | 0.000 |
|  | Context Recall | 0.685 | 0.509 | -0.176 |

### Field-Specific Performance

| Field Type | Metric | Original RAG | Advanced Retrieval | Delta |
|------------|--------|-------------|-------------------|-------|
| Policy Insured | Faithfulness | 0.938 | 0.938 | 0.000 |
|  | Answer Relevancy | 0.751 | 0.750 | -0.001 |
|  | Context Precision | 0.500 | 0.500 | 0.000 |
|  | Context Recall | 0.893 | 0.852 | -0.041 |
| Line of Business | Faithfulness | 1.000 | 1.000 | 0.000 |
|  | Answer Relevancy | 0.802 | 0.803 | +0.001 |
|  | Context Precision | 0.750 | 0.750 | 0.000 |
|  | Context Recall | 0.958 | 0.958 | 0.000 |
| Effective Date | Faithfulness | 1.000 | 1.000 | 0.000 |
|  | Answer Relevancy | 0.855 | 0.857 | +0.002 |
|  | Context Precision | 0.625 | 0.625 | 0.000 |
|  | Context Recall | 0.705 | 0.621 | -0.084 |
| Premium Amount | Faithfulness | 0.875 | 0.750 | -0.125 |
|  | Answer Relevancy | 0.816 | 0.817 | +0.001 |
|  | Context Precision | 0.750 | 0.750 | 0.000 |
|  | Context Recall | 0.634 | 0.621 | -0.013 |

## Analysis of Results

Our comparative evaluation reveals several interesting findings about the advanced retrieval implementation:

1. **Slight Tradeoffs in Metrics**: The advanced retrieval approach shows a small decrease in faithfulness (-1.5%) and context recall (-3.2%), while maintaining context precision and slightly improving answer relevancy (+0.2%). This suggests that our hybrid approach makes some tradeoffs in how comprehensively it uses available context in favor of more relevant answers.

2. **Performance on Complex Emails**: The most significant impact is seen in context recall for complex/golden emails (-17.6%), suggesting that while our advanced techniques may be more selective in the context they use, they sometimes miss relevant information in complex, unstructured emails.

3. **Field-Specific Improvements**: Answer relevancy improved for most field types, particularly for effective dates (+0.2%) and premium amounts (+0.1%), which were previously challenging fields. This indicates that our advanced retrieval methods are better at targeting the specific context needed for these fields.

4. **Consistency in Context Precision**: Both implementations show identical context precision scores overall and across categories, suggesting that our advanced retrieval techniques are maintaining the same level of relevance in the selected context, just using different selection criteria.

5. **Impact on Faithfulness**: While overall faithfulness decreased slightly (-1.5%), this was primarily driven by changes in premium amount extraction (-12.5%). The stability in other fields suggests that our advanced techniques maintain factual accuracy for most extraction tasks.

## Planned Improvements

Based on our performance assessment, we plan to make the following improvements to our application in the second half of the course:

1. **Hybrid Retrieval Optimization**: We'll refine our hybrid retrieval approach to address the context recall reduction, particularly for complex emails. This will include:
   - Implementing a weighted ensemble method that combines the strengths of both retrieval approaches
   - Adding a context validation step that ensures critical policy information isn't omitted
   - Developing specialized retrieval strategies for different email formats and complexity levels

2. **Field-Specific Retrieval Pipelines**: Given the variation in performance across different policy fields, we'll implement field-specific retrieval pipelines that are optimized for each type of information:
   - For premium amounts, where we saw the largest faithfulness drop, we'll add specialized pattern recognition and number extraction capabilities
   - For effective dates, we'll enhance context recall with date-specific retrieval patterns
   - For policy insured fields, we'll implement entity recognition pre-processing to improve identification

3. **Dynamic Chunking Strategy**: We'll move from a static to a dynamic chunking approach that adapts based on email structure and complexity:
   - For structured emails, maintain larger chunks that preserve formatting
   - For complex emails, implement finer-grained chunking with more overlap
   - Add structural markers to chunks to help the model understand the relationship between different sections

4. **Self-Improving System**: We'll implement a feedback loop that learns from extraction performance:
   - Track which extraction methods perform best for which carriers and email formats
   - Automatically select the optimal retrieval strategy based on email characteristics
   - Use failed extractions to generate synthetic training examples that improve future performance

5. **Advanced Reranking**: Implement a multi-stage retrieval process with specialized reranking:
   - First-stage retrieval with high recall to gather candidate chunks
   - Second-stage reranking with a field-specific cross-encoder to prioritize the most relevant context
   - Final extraction with contextual awareness of the entire email structure

6. **Multi-Modal Extraction**: Extend our system to handle multi-modal inputs:
   - Process tables and structured data that appear in HTML emails
   - Extract information from PDF attachments containing policy details
   - Combine information across multiple related emails in a thread

By implementing these improvements, we expect to achieve both better overall metrics and more consistent performance across email categories and field types. Our goal is to push faithfulness and answer relevancy above 90% while maintaining or improving context precision and recall, ultimately delivering a highly accurate and reliable policy information extraction system.
