from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import PyPDF2
import io
from src.rag_system import ResearchPaperRAG
import os
from src.visualizer import visualize_section
import uuid

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for documents and TLDRs
# Structure: {document_id: {"filename": str, "sections": [{"title": str, "text": str}], "tldrs": [{"status": str, "tldr": str or None}]}}
documents = {}

# Initialize RAG system (use a temp index for uploaded paper)
rag = ResearchPaperRAG("uploaded_paper_index")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a PDF, extract sections, and start async TLDR generation."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() or ""
        # Naive section split: split by double newlines or 'Section' keyword
        import re
        raw_sections = re.split(r"\n\n+|(?i)section ", full_text)
        sections = []
        for i, sec in enumerate(raw_sections):
            sec = sec.strip()
            if not sec:
                continue
            title = f"Section {i+1}"
            # Try to extract a title from the first line
            first_line = sec.split("\n", 1)[0]
            if len(first_line) < 80:
                title = first_line.strip()
            sections.append({"title": title, "text": sec})
        document_id = str(uuid.uuid4())
        documents[document_id] = {
            "filename": file.filename,
            "sections": sections,
            "tldrs": [{"status": "pending", "tldr": None} for _ in sections]
        }
        # Start TLDR generation in the background
        if background_tasks is not None:
            background_tasks.add_task(generate_tldrs_background, document_id)
        return {
            "document_id": document_id,
            "sections": [
                {"title": sec["title"], "index": i, "preview": sec["text"][:100]}
                for i, sec in enumerate(sections)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

def generate_tldrs_background(document_id: str):
    """Background task to generate TLDRs for all sections."""
    from src.rag_system import ResearchPaperRAG
    rag = ResearchPaperRAG("uploaded_paper_index")
    doc = documents[document_id]
    rag.papers = []
    rag.stored_data = []
    rag.index.reset()
    for i, section in enumerate(doc["sections"]):
        try:
            rag.add_to_index(section["text"], {"type": "section", "title": section["title"]})
            tldr = rag.generate_tldr(section["text"])
            doc["tldrs"][i]["status"] = "ready"
            doc["tldrs"][i]["tldr"] = tldr
        except Exception as e:
            doc["tldrs"][i]["status"] = "error"
            doc["tldrs"][i]["tldr"] = str(e)

@app.get("/tldr")
async def get_tldrs(document_id: str):
    """Poll for TLDRs for all sections of a document."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc = documents[document_id]
    return {
        "tldrs": [
            {"title": sec["title"], "status": tldr["status"], "tldr": tldr["tldr"]}
            for sec, tldr in zip(doc["sections"], doc["tldrs"])
        ]
    }

@app.post("/qna")
async def ask_question(document_id: str = Form(...), question: str = Form(...)):
    """Ask a question about the uploaded paper."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc = documents[document_id]
    # Ensure the RAG index is up to date with the current sections
    rag.papers = []
    rag.stored_data = []
    rag.index.reset()
    for section in doc["sections"]:
        rag.add_to_index(section["text"], {"type": "section", "title": section["title"]})
    # Build context as a dict: {title: text}
    context = {section["title"]: section["text"] for section in doc["sections"]}
    answer = rag.answer_question(question, context)
    return {"answer": answer}

@app.get("/visualize")
async def get_visualization(document_id: str, section_title: Optional[str] = None):
    """Get a visualization for a section."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc = documents[document_id]
    # Find the section by title (default to first section if not specified)
    section = None
    if section_title:
        for sec in doc["sections"]:
            if sec["title"] == section_title:
                section = sec
                break
    if not section:
        section = doc["sections"][0]
    # Generate visualization
    image_path = visualize_section(section["title"], section["text"])
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=500, detail="Failed to generate visualization.")
    # Return the image file
    return FileResponse(image_path, media_type="image/png")

@app.get("/sections")
async def get_sections(document_id: str):
    """Get a list of all section titles in the uploaded PDF."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc = documents[document_id]
    return {
        "sections": [
            {"title": sec["title"], "index": i, "preview": sec["text"][:100]}
            for i, sec in enumerate(doc["sections"])
        ]
    }
