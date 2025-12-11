import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

VECTOR_FILE = "vectorstore.pkl"
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------------
# Extract text from PDF
# -------------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
    return full_text

# -------------------------------------
# Split text into chunks
# -------------------------------------
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------------------------
# Process PDF and save embeddings
# -------------------------------------
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump({"documents": chunks, "embeddings": embeddings}, f)
    return True

# -------------------------------------
# Load embeddings
# -------------------------------------
def load_vectorstore():
    if not os.path.exists(VECTOR_FILE):
        return None, None
    with open(VECTOR_FILE, "rb") as f:
        data = pickle.load(f)
    return data["documents"], data["embeddings"]

# -------------------------------------
# Cosine similarity search
# -------------------------------------
def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)

# -------------------------------------
# Ask question
# -------------------------------------
def ask_question(query, top_k=3):
    documents, document_embeddings = load_vectorstore()
    if documents is None or document_embeddings is None:
        return "No PDF uploaded yet. Please upload one first."
    
    query_emb = embedding_model.encode([query], convert_to_numpy=True)[0]
    scores = cosine_similarity(document_embeddings, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    result = "\n".join([documents[i] for i in top_indices])
    
    if not result.strip():
        return "No relevant content found."
    return result
