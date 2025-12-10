from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings()
    )
    return vectordb

def ask_question(vectordb, query):
    llm = OpenAI(temperature=0)

    docs = vectordb.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Use ONLY the context to answer:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(prompt)

    return answer
