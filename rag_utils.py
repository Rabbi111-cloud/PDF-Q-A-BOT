import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------------------------------------
# GLOBAL VARIABLES
# -------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = []          # Stores text chunks
document_embeddings = None  # NumPy matrix
faiss_index = None      # The FAISS index


# -------------------------------------
# 1. Extract text from PDF
# -------------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"

    return full_text


# -------------------------------------
# 2. Split text into chunks
# -------------------------------------
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------------
# 3. Build FAISS index from chunks
# -------------------------------------
def process_pdf(pdf_path):
    global documents, document_embeddings, faiss_index

    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    documents = chunks

    # Embed all chunks
    document_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    # Build FAISS index
    dim = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(document_embeddings)

    print("PDF processed and indexed successfully.")


# -------------------------------------
# 4. Search & Answer questions
# -------------------------------------
def ask_question(query, top_k=3):
    global documents, document_embeddings, faiss_index

    if faiss_index is None:
        return "Please upload a PDF first."

    # Embed the question
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Search similar chunks
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_chunks = [documents[i] for i in indices[0] if i < len(documents)]

    # Combine into simple answer
    combined_text = "\n".join(retrieved_chunks)

    if combined_text.strip() == "":
        return "No relevant information found in the PDF."

    return combined_text
