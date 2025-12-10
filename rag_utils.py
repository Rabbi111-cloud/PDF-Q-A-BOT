import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

VECTOR_DB_DIR = "vectordb"

# Ensure directory exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ---------------------------------
# Load existing vectorstore if any
# ---------------------------------
def load_vectorstore():
    """Return a Chroma vectorstore instance if DB exists, otherwise None."""
    if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return None


# ---------------------------------
# Process & index PDF
# ---------------------------------
def process_pdf(pdf_path: str):
    """Load PDF, split into chunks, embed, and store in Chroma DB."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vectordb.persist()

    return True


# ---------------------------------
# Ask question from indexed PDF
# ---------------------------------
def ask_question(query: str):
    """Retrieve relevant chunks & ask LLM."""
    vectordb = load_vectorstore()
    if vectordb is None:
        return "No PDF has been uploaded yet."

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)

    content = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    You are a helpful AI assistant. Answer the question using ONLY the text below.
    If the answer is not found, say "I cannot find the answer in the PDF."

    PDF TEXT:
    {content}

    QUESTION: {query}
    """

    response = llm.invoke(prompt)
    return response.content
