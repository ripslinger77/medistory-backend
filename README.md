# Medistory Function App

A serverless Azure Functions application that processes medical transcripts and generates SOAP notes using Azure OpenAI. The application provides three main functionalities: generating clinical notes from transcripts, summarizing patient data, and answering questions about patient history.

## Features

- **Generate SOAP Notes**: Converts doctor-patient conversation transcripts into structured SOAP (Subjective, Objective, Assessment, Plan) notes
- **Patient Data Summarization**: Creates concise summaries of patient history from SOAP notes
- **Medical Q&A**: Answers questions about patient data using conversational AI with RAG (Retrieval Augmented Generation)

## Architecture

This application uses:
- **Azure Functions** (Python 3.11) for serverless compute
- **Azure OpenAI** for LLM capabilities (GPT-4o-mini)
- **Azure Blob Storage** for storing transcripts and generated notes
- **LangChain** for building LLM workflows and RAG chains
- **FAISS** for vector storage and semantic search

## Project Structure

```
.
├── generatenote/          # Function: Generate SOAP notes from transcripts
│   ├── __init__.py
│   └── function.json
├── summary/               # Function: Summarize patient data
│   ├── __init__.py
│   └── function.json
├── qa/                    # Function: Q&A about patient history
│   ├── __init__.py
│   └── function.json
├── modules/
│   ├── config.py         # Azure OpenAI configuration
│   └── llm.py            # LLM operations and blob storage utilities
├── .github/workflows/    # GitHub Actions for CI/CD
├── host.json
└── requirements.txt
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd medistory-function-app
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `local.settings.json` file for local development:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AZURE_OPENAI_API_KEY": "your-openai-api-key",
    "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/",
    "AZURE_OPENAI_EMBEDDING_KEY": "your-embedding-api-key",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-mini",
    "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    "EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-small",
    "EMBEDDING_API_VERSION": "2024-02-01",
    "AZURE_STORAGE_CONNECTION_STRING": "your-storage-connection-string",
    "TRANSCRIPT_CONTAINER": "conversation-transcripts",
    "GEN_NOTE_CONTAINER": "generated-notes",
    "SOAP_NOTES_CONTAINER": "patient-history"
  }
}
```

### 4. Run Locally

```bash
func start
```

## API Endpoints

### 1. Generate SOAP Note

**Endpoint**: `POST /api/generatenote`

**Description**: Generates a SOAP note from a patient transcript stored in Azure Blob Storage.

**Request Body**:
```json
{
  "patient_id": "12345"
}
```

**Response**:
```json
{
  "patient_id": "12345",
  "soap_note": "Subjective:\n...\n\nObjective:\n...\n\nAssessment:\n...\n\nPlan:\n..."
}
```

**Storage Requirements**:
- Input: Transcript file at `transcript_{patient_id}.txt` in the `conversation-transcripts` container
- Output: Generated note saved to `generated_soap_note_{patient_id}.json` in the `generated-notes` container

### 2. Summarize Patient Data

**Endpoint**: `POST /api/summary`

**Description**: Creates a concise summary of patient history from an existing SOAP note.

**Request Body**:
```json
{
  "patient_id": "12345"
}
```

**Response**:
```json
{
  "patient_id": "12345",
  "summary": "- Chief Complaint:\n  - ...\n\n- History of Present Illness:\n  - ..."
}
```

**Storage Requirements**:
- Input: SOAP note at `soap_note_{patient_id}.json` in the `patient-history` container

### 3. Question & Answer

**Endpoint**: `POST /api/qa`

**Description**: Answers questions about patient data using conversational AI.

**Request Body**:
```json
{
  "patient_id": "12345",
  "question": "What medications is the patient currently taking?",
  "chat_history": []
}
```

**Response**:
```json
{
  "answer": "Based on the patient's records, they are currently taking...",
  "chat_history": [
    ["What medications is the patient currently taking?", "Based on the patient's records, they are currently taking..."]
  ]
}
```

**Storage Requirements**:
- Input: SOAP note at `soap_note_{patient_id}.json` in the `patient-history` container

## Deployment

### Deploy to Azure using GitHub Actions

This project includes a GitHub Actions workflow for automated deployment.

1. Set up GitHub secrets:
   - `AZUREAPPSERVICE_CLIENTID_*`
   - `AZUREAPPSERVICE_TENANTID_*`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_*`

2. Push to the `main` branch to trigger deployment:
```bash
git push origin main
```

### Manual Deployment

```bash
# Login to Azure
az login

# Deploy function app
func azure functionapp publish medistory-function-app
```

### Configure Application Settings

After deployment, configure the environment variables in Azure Portal:

1. Navigate to your Function App
2. Go to Configuration → Application settings
3. Add all environment variables from `local.settings.json`

## Azure Blob Storage Structure

### Required Containers

1. **conversation-transcripts**: Stores patient conversation transcripts
   - Format: `transcript_{patient_id}.txt`

2. **generated-notes**: Stores generated SOAP notes
   - Format: `generated_soap_note_{patient_id}.json`

3. **patient-history**: Stores patient SOAP notes for querying
   - Format: `soap_note_{patient_id}.json`

## Development

### Running Tests

```bash
# Add test step in workflow or run locally
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Format code with:

```bash
black .
flake8 .
```

## Monitoring

Monitor function execution through:
- Azure Portal → Function App → Monitor
- Application Insights (enabled by default)
- Function logs: `func logs --follow`

## Security Considerations

- All endpoints use function-level authentication (`authLevel: "function"`)
- API keys are stored as environment variables
- Storage connection strings should use managed identities in production
- Ensure HIPAA compliance for patient data storage
