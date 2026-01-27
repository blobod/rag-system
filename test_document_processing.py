from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=== Document Processing Test ===\n")

# Load PDF
print("1. Loading PDF")
loader = PyPDFLoader("test_document.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s)")
print(f"Total content length: {sum(len(doc.page_content) for doc in documents)} characters\n")

# Split into chunks
print("2. Splitting into chunks")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks\n")

# Display first few chunks
print("3. Preview of first 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1} (page {chunk.metadata.get('page', 'unknown')}):\n")
    print(f"{chunk.page_content[:150]}\n")
    
print("=== Test Complete ===")