# Enterprise Document Intelligence System

An intelligent document Q&A system using RAG (Retrieval-Augmented Generation) that ingests PDF documents and provides accurate answers using LLMs with retrieved context.

**Status:** Production Ready

**Live Demo:** http://rag-system-api.westeurope.azurecontainer.io:8000

## Progress

- ✅ Docker environment with GPU support
- ✅ RAG pipeline with Mistral-7B and LangChain
- ✅ FastAPI REST API
- ✅ Error handling and logging
- ✅ Azure deployment

## Technology Stack

**Core Technologies**

- Python
- Docker with NVIDIA GPU support
- FastAPI
- LangChain

**ML/AI**

- Mistral-7B-Instruct (LLM)
- sentence-transformers (embeddings)
- ChromaDB (vector database)

## Setup

### Prerequisites

- Docker Desktop installed
- NVIDIA GPU with 6GB+ VRAM
- WSL2 (Windows) with Ubuntu
- NVIDIA Container Toolkit
- Hugging Face account (for model downloads)

### Installation

1. **Clone the repository**

```
git clone <your-repo-url>
cd rag-system
```

2. **Create .env file** (optional, for faster downloads)

```
HF_TOKEN=your_huggingface_token_here
```

3. **Build the Docker container**

```
docker-compose build
```

This will automatically install all dependencies from `requirements.txt`.

4. **Start the container**

```
docker-compose run --rm -p 8000:8000 rag-dev
```

5. **Run the API** (inside container)

```
python3 app.py
```

The server will start and load models (takes few minutes).

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Usage

### API Endpoints

Access the interactive API documentation at: http://localhost:8000/docs

**1. Health Check**

```
curl http://localhost:8000/
```

**2. Upload PDF Document**

```
curl -X POST "http://localhost:8000/upload" -F "file=@your_document.pdf"
```

**3. Ask Questions**

```
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"Your question here\", \"k\": 2}"
```

Parameters:

- `question`: Your question about the uploaded documents
- `k`: Number of relevant chunks to retrieve (1-10, default: 2)

### Example

```
# Upload a document
curl -X POST "http://localhost:8000/upload" -F "file=@test_document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is machine learning?\"}"
```

## Azure Deployment

The system is deployed on Azure Container Instances at:
**http://rag-system-api.westeurope.azurecontainer.io:8000**

### Performance Note

**Important:** The Azure deployment runs on CPU due to GPU limitations. This results in slower inference times (~15-20 minutes per query, which results in the process timing out).

**Performance Comparison:**

- Local with GPU (RTX 3080 Ti): ~2-3 minutes per query
- Azure with CPU: ~15-20 minutes per query, which results in a fail

## Testing

Several test scripts are available to test individual components:

```
# Test GPU and CUDA
python3 test_gpu.py

# Test Mistral-7B LLM
python3 test_mistral.py

# Test document processing
python3 test_document_processing.py

# Test vector store
python3 test_vector_store.py

# Test complete RAG pipeline
python3 test_rag_pipeline.py
```

## Logging

Application logs are written to:

- Console output
- `rag_api.log` file

Logs include detailed information about:

- Document processing
- Vector store operations
- Model inference
- Retrieved context for each query
- Errors and warnings

## Project Structure

```
rag-system/
├── app.py                          # FastAPI application
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Docker orchestration
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not in git)
├── .gitignore                      # Git ignore rules
├── tests/
│   ├── test_mistral.py             # LLM test
│   ├── test_document_processing.py
│   ├── test_vector_store.py
│   └── test_rag_pipeline.py
├── test_document.pdf               # Sample documents
├── test_document2.pdf
├── chroma_db/                      # Vector database (persisted)
└── rag_api.log                     # Application logs
```

## Architecture

1. **Document Upload:** PDFs are loaded, split into chunks (500 chars, 100 overlap), and embedded using sentence-transformers
2. **Vector Storage:** Embeddings stored in ChromaDB for semantic search
3. **Query Processing:** User questions are embedded and matched against document chunks
4. **Answer Generation:** Top-k relevant chunks are retrieved and passed to Mistral-7B with the question
5. **Response:** Generated answer returned with source attribution

## This is just a simple portflio project
