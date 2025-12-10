from fastapi import FastAPI, UploadFile, File, Form
from rag_utils import process_pdf, ask_question
import tempfile

app = FastAPI()

vectordb = None

@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    global vectordb

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await pdf.read())
    temp.close()

    vectordb = process_pdf(temp.name)
    return {"message": "PDF uploaded and processed."}

@app.post("/ask")
async def ask(q: str = Form(...)):
    global vectordb
    if vectordb is None:
        return {"error": "Upload PDF first"}

    answer = ask_question(vectordb, q)
    return {"answer": answer}
