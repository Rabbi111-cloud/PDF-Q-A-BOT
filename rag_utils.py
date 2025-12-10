import os
"""Load PDF, split, embed, and persist Chroma DB to CHROMA_DIR."""
loader = PyPDFLoader(pdf_path)
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()


vectordb = Chroma.from_documents(
documents=chunks,
embedding=embeddings,
persist_directory=CHROMA_DIR,
)


# Persist to disk
vectordb.persist()




def get_vectordb() -> Optional[Chroma]:
"""Return a Chroma vectorstore instance if DB exists, otherwise None."""
# If the persist dir exists and has files, return a live Chroma instance
if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
embeddings = OpenAIEmbeddings()
return Chroma(persist_directory=CHROMA_DIR, embedding=embeddings)
return None




def ask_question(query: str) -> str:
"""Run retrieval and call the LLM. Returns string answer or an error message."""
vectordb = get_vectordb()
if vectordb is None:
return "Error: No PDF processed yet. Please upload and process a PDF first."


docs = vectordb.similarity_search(query, k=4)
if not docs:
return "No relevant information found in the processed PDFs."


context = "\n\n".join([d.page_content for d in docs])


llm = OpenAI(temperature=0)


prompt = f"""
You are given the following context extracted from a PDF. Use ONLY this context to answer the user's question. If the answer is not contained in the context, say you don't know.


Context:
{context}


Question: {query}


Answer:
"""


response = llm.invoke(prompt)


# Ensure it's a string
if hasattr(response, "text"):
return response.text
return str(response)
