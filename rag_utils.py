import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

VECTOR_DB_DIR = "vectordb"

# Ensure vector DB directory exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# -------------------------------
# Load VectorStore If Exists
# -------------------------------
def load_vectorstore():
    """Load Chroma DB if available."""
    if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return None


# -------------------------------
# Process Uploaded PDF
# -------------------------------
def process_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    return True


# -------------------------------
# Ask Question From VectorStore
# -------------------------------
def ask_question(query: str):
    vectordb = load_vectorstore()
    if vectordb is None:
        return "No PDF uploaded yet. Please upload one first."

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    Answer the question using ONLY the text below.
    If the answer cannot be found in the document, say so.

    CONTEXT:
    {context}

    QUESTION: {query}
    """

    response = llm.invoke(prompt)
    return response.content
