from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# Initialize FastAPI
app = FastAPI(
    title="RAG Document Intelligence API",
    description="Upload PDFs and ask questions using RAG",
    version="1.0.0"
)

#global variables for models
embeddings = None
vectorstore = None
llm = None
retriever = None
rag_chain = None

# Request/Response and Upload/Error models
class QuestionRequest(BaseModel):
    question: str
    k: int = 2  # default numbers of chunks to retrieve

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources_count: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int

class ErrorResponse(BaseModel):
    error: str
    detail: str

@app.on_event("startup")
async def startup_event():
    """Load models when server starts"""
    global embeddings, llm
    
    try:
        logger.info("Starting RAG API server")
        
        logger.info("Loading embedding model")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Embedding model loaded successfully")
        
        logger.info("Loading Mistral-7B (may take a few minutes)")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Mistral-7B loaded")
        
        logger.info("RAG API server started")
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {
        "status": "running",
        "message": "RAG Document Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "upload": "/upload (POST)",
            "ask": "/ask (POST)",
            "docs": "/docs"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    global vectorstore, retriever, rag_chain

    logger.info(f"Upload requested: {file.filename}")
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        logger.info(f"Saving file to: {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load document
        logger.info("Loading PDF...")
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        logger.info("Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create/update vector store
        logger.info("Adding to vector store...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("Created new vector store")
        else:
            vectorstore.add_documents(chunks)
            logger.info("Added to existing vector store")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Create RAG chain
        logger.info("Creating RAG chain...")
        template = """<s>[INST] You must ONLY use the following context to answer questions. 
        Do NOT use any external knowledge. 
        If the answer is not in the context, respond with exactly: "I cannot answer this based on the provided documents."

        Context:
        {context}

        Question: {question} [/INST]

        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain created")
        
        logger.info(f"Successfully processed {file.filename}")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            pages=len(documents),
            chunks=len(chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    
    logger.info(f"Question received: {request.question} (k={request.k})")
    
    try:
        # Check if documents are uploaded
        if vectorstore is None or rag_chain is None:
            logger.warning("Question asked but no documents uploaded")
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload a document first."
            )
        
        # Validate k parameter
        if request.k < 1 or request.k > 10:
            logger.warning(f"Invalid k value: {request.k}")
            raise HTTPException(
                status_code=400,
                detail="k must be between 1 and 10"
            )
        
        # Get retriever with custom k
        logger.info(f"Retrieving top {request.k} relevant chunks...")
        retriever_with_k = vectorstore.as_retriever(search_kwargs={"k": request.k})
        
        # Get relevant documents
        relevant_docs = retriever_with_k.invoke(request.question)
        logger.info(f"Retrieved {len(relevant_docs)} chunks")
        
        # Generate answer with custom retriever
        logger.info("Generating answer...")
        template = """<s>[INST] Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question} [/INST]

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain_dynamic = (
            {"context": retriever_with_k | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate answer
        answer = rag_chain_dynamic.invoke(request.question)
        logger.info("Answer generated successfully")
        
        return AnswerResponse(
            question=request.question,
            answer=answer.strip(),
            sources_count=len(relevant_docs)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)