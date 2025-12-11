import os
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

VECTOR_DB_DIR = "vectordb"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# -----------------------------
# Load VectorStore
# -----------------------------
def load_vectorstore():
    if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return None

# -----------------------------
# Process PDF
# -----------------------------
def process_pdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vectordb.persist()
        return True
    except Exception as e:
        print("❌ Error in process_pdf:", e)
        raise e

# -----------------------------
# Ask Question
# -----------------------------
def ask_question(query: str):
    try:
        vectordb = load_vectorstore()
        if vectordb is None:
            return "No PDF uploaded yet. Please upload one first."

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([d.page_content for d in docs])
        if not context.strip():
            return "No relevant content found in PDF."

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
    except Exception as e:
        print("❌ Error in ask_question:", e)
        return "Error while processing your question."

