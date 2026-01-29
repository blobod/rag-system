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

# Initialize FastAPI
app = FastAPI(title="RAG Document Intelligence API")

#global variables for models
embeddings = None
vectorstore = None
llm = None
retriever = None
rag_chain = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    k: int = 2  # default numbers of chunks to retrieve

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources_count: int

@app.on_event("startup")
async def startup_event():
    """Load models when server starts"""
    global embeddings, llm
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("Loading Mistral-7B...")
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
    print("Models loaded successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "RAG Document Intelligence API",
        "endpoints": {
            "upload": "/upload (POST)",
            "ask": "/ask (POST)"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    global vectorstore, retriever, rag_chain
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load document
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create/update vector
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vectorstore.add_documents(chunks)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Create RAG
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
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    
    if vectorstore is None or rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Please upload a document first."
        )
    
    # Update retriever
    retriever_with_k = vectorstore.as_retriever(search_kwargs={"k": request.k})
    
    # Get relevant documents
    relevant_docs = retriever_with_k.invoke(request.question)
    
    # Generate answer with custom retriever
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
    
    return AnswerResponse(
        question=request.question,
        answer=answer.strip(),
        sources_count=len(relevant_docs)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)