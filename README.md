# Insurance Policy Information Extraction System

## Overview

This system automates the extraction of critical policy information from insurance-related emails. It uses advanced AI techniques to identify and extract key details such as:

- Insured party/policyholder names
- Line of business/policy types
- Effective dates
- Inception dates
- Premium amounts

The system is designed to integrate with the Kitsune insurance platform to streamline underwriters' workflows by reducing manual data entry and improving accuracy.

## Key Features

- **Automated Email Processing**: Processes insurance emails to extract structured policy data
- **Hybrid Retrieval System**: Combines dense and sparse retrieval techniques for optimal information extraction
- **High Accuracy**: Achieves strong performance metrics for faithfulness (87.5%) and answer relevancy (81.2%)
- **Field-Specific Extraction**: Specialized extraction techniques for different policy field types
- **Evaluation Framework**: Comprehensive RAGAS-based evaluation for continuous improvement

## Technology Stack

- **LLM**: GPT-4 (OpenAI API)
- **Embedding Models**: OpenAI's text-embedding-3-small
- **Orchestration**: LangGraph with FastAPI
- **Vector Database**: Qdrant
- **Monitoring**: LangSmith
- **Evaluation**: RAGAS framework
- **Containerization**: Docker Compose

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key
- Gmail API credentials (for email access)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/aie-certification-challenge.git
   cd aie-certification-challenge
   ```

2. Create a .env file:
   ```
   cp .env.template .env
   ```
   Then edit the .env file to add your API keys and configuration values.

3. Build and start services:
   ```
   make build
   make up
   ```

### Configuration

1. Place your Google API credentials in the root directory:
   - `google_credentials.json`: Service account credentials
   - `gmail_token.json`: OAuth token for Gmail access

2. Configure settings in `src/config/settings.py` as needed

## Usage

### Running the API Service

Start the API service:
```
make api
```

The API will be available at http://localhost:8000

### API Endpoints

The API includes Swagger documentation for easy testing and interaction. Once the API is running, visit:

```
http://localhost:8000/docs
```

This provides an interactive UI where you can test all endpoints without needing curl commands.

Main endpoints include:

- **POST /process-email**: Process a new email
- **GET /policy/{policy_id}**: Retrieve extracted policy information

### Running Evaluations

Evaluate the system on the golden dataset:
```
make evaluate
```

## Project Structure

- `src/`: Core application code
  - `api/`: FastAPI endpoints
  - `config/`: Configuration settings
  - `models/`: Data models
  - `rag/`: RAG implementation and LangGraph workflow
  - `services/`: Core services (email processing, embedding, retrieval)
- `evaluations/`: Evaluation code and results
- `tests/`: Test suite (unit, integration, e2e)

## Performance

| Metric | Score |
|--------|-------|
| Faithfulness | 87.5% |
| Answer Relevancy | 81.2% |
| Context Precision | 62.5% |
| Context Recall | 73.4% |

For detailed performance analysis and planned improvements, see the [DELIVERABLES.md](./DELIVERABLES.md) document.

## Testing

Run the test suite:
```
make test
```

Run specific test types:
```
make unit-tests
make integration-tests
make e2e-tests
```

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature-name`
2. Implement changes
3. Add tests
4. Run existing tests: `make test`
5. Create pull request

### Makefile Commands

- `make build`: Build Docker containers
- `make up`: Start all services
- `make down`: Stop all services
- `make api`: Start API service
- `make test`: Run all tests
- `make evaluate`: Run evaluation on golden dataset
