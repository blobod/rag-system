from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("=== Vector Store Test ===\n")

# Load and chunk document
print("1. Loading and chunking PDF")
loader = PyPDFLoader("test_document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks\n")

# Create embeddings
print("2. Loading embedding model")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded\n")

# Create vector store
print("3. Creating vector store and adding documents")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Stored {len(chunks)} chunks in ChromaDB\n")

# Test retrieval
print("4. Testing retrieval")
query = "What is deep learning?"
results = vectorstore.similarity_search(query, k=2)

print(f"Query: '{query}'")
print(f"Found {len(results)} relevant chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"{doc.page_content[:200]}\n")

print("=== Test Complete ===")