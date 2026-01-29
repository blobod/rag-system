# Enterprise Document Intelligence System

An intelligent document Q&A system using RAG (Retrieval-Augmented Generation) that ingests company documents and provides accurate answers using LLMs with retrieved context.

**Status:** In Development

## Current Planned Technology Stack

**Core Technologies**

- Python
- Docker
- Azure
- Databricks
- FastAPI
- LangChain (RAG Framework)

## Setup

### Prerequisites

- Docker Desktop installed
- NVIDIA GPU with 6GB+ VRAM
- WSL2 (Windows) with Ubuntu
- NVIDIA Container Toolkit

### Build and Run

0. Create .env file that contains HF_Token
   This should be the content of the .env:

```
HF_TOKEN=YourToken
```

1. Build the Docker container:

```
docker-compose build
```

2. Start the container:

```
docker-compose run --rm rag-dev
```

3. Inside the container, install dependencies:

```
pip3 install transformers sentence-transformers torch accelerate bitsandbytes langchain langchain-community langchain-core pypdf chromadb fastapi uvicorn python-multipart
```

### Test the Setup

Run the Mistral-7B test:

```
python3 test_mistral.py
```

This will:

- Download Mistral-7B-Instruct (~13GB, first run only)
- Load with 8-bit quantization
- Generate a test response

Expected output: A one-sentence answer about machine learning.
