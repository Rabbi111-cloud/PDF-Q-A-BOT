from fastapi import FastAPI, UploadFile, File, Form
from rag_utils import process_pdf, ask_question
import tempfile

app = FastAPI()

# ---------------------------
# Upload and Index PDF
# ---------------------------
@app.post("/upload")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf_file.read())
            tmp_path = tmp.name

        # Process + Index
        process_pdf(tmp_path)
        return {"message": "PDF uploaded and processed successfully"}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# Ask Question Endpoint
# ---------------------------
@app.post("/ask")
async def ask(query: str = Form(...)):
    try:
        answer = ask_question(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
