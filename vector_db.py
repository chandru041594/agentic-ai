from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

def build_vector_store(texts):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(texts)
    db = FAISS.from_embeddings(vectors, texts)
    return db
