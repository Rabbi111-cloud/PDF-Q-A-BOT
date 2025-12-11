import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# -------------------------------
# VECTORSTORE DIRECTORY
# -------------------------------
VECTOR_DB_DIR = "vectordb"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# -------------------------------
# Load VectorStore if it exists
# -------------------------------
def load_vectorstore():
    if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return None

# -------------------------------
# Process PDF and index chunks
# -------------------------------
def process_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Embed chunks
    embeddings = OpenAIEmbeddings()

    # Create or update Chroma DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    return True

# -------------------------------
# Ask a question using the vectorstore
# -------------------------------
def ask_question(query: str):
    vectordb = load_vectorstore()
    if vectordb is None:
        return "No PDF uploaded yet. Please upload one first."

    # Retrieve relevant chunks
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    docs = retriever.get_relevant_documents(query)

    # Combine chunks into context
    context = "\n\n".join([d.page_content for d in docs])

    # Generate answer using ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-3.5-turbo"

    prompt = f"""
Answer the question using ONLY the text below.
If the answer cannot be found in the document, say so.

CONTEXT:
{context}

QUESTION: {query}
"""

    response = llm.invoke(prompt)
    return response.content

      
