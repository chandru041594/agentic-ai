from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Example documents (replace with your own text or PDF ingestion)
docs = ["Your knowledge base text here..."]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(docs)

# Generate embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(chunks)

# Build FAISS vector store
db = FAISS.from_embeddings(vectors, chunks)
