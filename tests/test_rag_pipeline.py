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

print("=== RAG Pipeline Test ===\n")
print("1. Setting up vector store")
loader = PyPDFLoader("test_document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"Vector store ready with {len(chunks)} chunks\n")

# Loading model
print("2. Loading Mistral-7B")
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
print("Mistral loaded\n")

# Create RAG chain using LangChain Expression Language
print("3. Creating RAG chain")

template = """<s>[INST] Use the following context to answer the question. If you cannot answer based on the context, say so.

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

print("RAG chain ready\n")
print("4. Testing RAG with questions\n")

questions = [
    "What is deep learning?",
    "What are the types of machine learning?"
]

for i, question in enumerate(questions, 1):
    print(f"Question {i}: {question}")
    
    # Get relevant documents first
    relevant_docs = retriever.invoke(question)
    
    # Generate answer
    answer = rag_chain.invoke(question)
    
    print(f"Answer: {answer.strip()}\n")
    print(f"Sources used: {len(relevant_docs)} chunks")
    print("-" * 50 + "\n")

print("=== RAG Pipeline Test Complete ===")