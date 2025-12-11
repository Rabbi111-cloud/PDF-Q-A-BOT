from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_utils import process_pdf, ask_question
import tempfile

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/process")
async def process_endpoint(pdf: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            tmp_path = tmp.name

        process_pdf(tmp_path)
        return {"message": "PDF processed successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_endpoint(payload: dict):
    try:
        query = payload.get("question", "")
        answer = ask_question(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

