import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------------------------------------
# GLOBAL VARIABLES
# -------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = []               # Stores text chunks
document_embeddings = None   # Stores embeddings as numpy array


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
# 3. Process PDF and embed chunks
# -------------------------------------
def process_pdf(pdf_path):
    global documents, document_embeddings

    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    documents = chunks
    document_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    print("PDF processed successfully.")


# -------------------------------------
# 4. Cosine similarity search
# -------------------------------------
def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)


# -------------------------------------
# 5. Ask question
# -------------------------------------
def ask_question(query, top_k=3):
    global documents, document_embeddings

    if document_embeddings is None:
        return "Please upload a PDF first."

    # Embed question
    query_emb = embedding_model.encode([query], convert_to_numpy=True)[0]

    # Compute cosine similarity for all chunks
    scores = cosine_similarity(document_embeddings, query_emb)

    # Get top_k chunk indexes
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Combine results into answer
    result = "\n".join([documents[i] for i in top_indices])

    if not result.strip():
        return "No relevant content found."

    return result

