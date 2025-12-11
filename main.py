from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_utils import process_pdf, ask_question
import tempfile

app = FastAPI()

# Enable CORS so your frontend (Gradio) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# Process + Index PDF
# ---------------------------
@app.post("/process")
async def process_endpoint(pdf: UploadFile = File(...)):
   
