from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_utils import process_pdf, ask_question
import tempfile
import os

app = FastAPI(title="PDF QnA Bot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf_file.read())
            tmp_path = tmp.name
        process_pdf(tmp_path)
        os.remove(tmp_path)
        return {"message": "PDF uploaded and processed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_endpoint(query: str = Form(...)):
    try:
        answer = ask_question(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

