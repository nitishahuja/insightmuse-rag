from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import io
from src.rag_system import ResearchPaperRAG
import os
import uuid
from src.pdf_extractor import extract_sections_from_pdf
from src.rag_pipeline import simplify_section
import asyncio
import time

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
# Structure: {document_id: {"filename": str, "sections": {section_name: section_text}, "images": [], "tldrs": {}, "processing_status": str}}
documents = {}

# Initialize RAG system (use a temp index for uploaded paper)
rag = ResearchPaperRAG("index")  # Use the main index with scitldr data

@app.post("/upload_pdf")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a PDF, extract sections, and start async TLDR generation."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        # Read file content
        content = await file.read()
        
        # Save to a temporary file since pdf_extractor expects a file path
        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        try:
            # Extract sections and images using the robust extractor
            sections_dict, images = extract_sections_from_pdf(temp_path)
            
            # Convert sections dict to list format
            sections = [
                {
                    "title": title,
                    "text": text,
                    "word_count": len(text.split())
                }
                for title, text in sections_dict.items()
            ]
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Store sections and prepare for TLDR generation
            documents[doc_id] = {
                "filename": file.filename,
                "sections": sections,
                "images": images,
                "tldrs": {},
                "processing_status": "processing"
            }
            
            # Start background TLDR generation
            if background_tasks is not None:
                background_tasks.add_task(generate_tldrs_for_document, doc_id)
            
            # Return in the standardized format
            return {
                "document_id": doc_id,
                "filename": file.filename,
                "total_sections": len(sections),
                "sections": [
                    {
                        "index": i,
                        "title": section["title"],
                        "text": section["text"],
                        "word_count": section["word_count"],
                        "preview": section["text"][:200] + "..." if len(section["text"]) > 200 else section["text"]
                    }
                    for i, section in enumerate(sections)
                ]
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

async def generate_tldrs_for_document(doc_id: str):
    try:
        doc = documents[doc_id]
        sections = doc["sections"]
        
        for section in sections:
            try:
                # Use the RAG-based TLDR generation
                tldr = rag.generate_tldr(section["text"])
                doc["tldrs"][section["title"]] = {
                    "content": tldr,
                    "status": "completed"
                }
            except Exception as e:
                doc["tldrs"][section["title"]] = {
                    "content": str(e),
                    "status": "error"
                }
                
        doc["processing_status"] = "completed"
    except Exception as e:
        documents[doc_id]["processing_status"] = "error"
        print(f"Error generating TLDRs for document {doc_id}: {str(e)}")

@app.get("/tldr")
async def get_tldr(document_id: str, wait: bool = True):
    """Get TLDR for a document with optional wait for processing."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[document_id]
    
    # If wait is True, wait for processing to complete
    if wait:
        timeout = 60  # 60 second timeout
        start_time = time.time()
        while doc["processing_status"] == "processing":
            if time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="TLDR generation timeout")
            await asyncio.sleep(1)
    
    # Combine sections and their TLDRs into a single response
    sections_with_tldr = []
    for section in doc["sections"]:
        title = section["title"]
        tldr_info = doc["tldrs"].get(title, {
            "content": None,
            "status": "pending" if doc["processing_status"] == "processing" else "error"
        })
        
        sections_with_tldr.append({
            "title": title,
            "content": section["text"],
            "word_count": section["word_count"],
            "tldr": tldr_info["content"],
            "status": tldr_info["status"]
        })

    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "processing_status": doc["processing_status"],
        "total_sections": len(doc["sections"]),
        "sections": sections_with_tldr
    }

@app.post("/qna")
async def ask_question(document_id: str = Form(...), question: str = Form(...)):
    """Ask a question about the uploaded paper."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    doc = documents[document_id]
    
    # Build context from the paper sections
    context = {section["title"]: section["text"] for section in doc["sections"]}
    
    try:
        # Get answer using the RAG system
        answer = rag.answer_question(question, context)
        
        # Get relevant research context
        relevant_context = rag.get_relevant_context(question, k=3)
        
        # Return both the answer and relevant context
        return {
            "answer": answer,
            "relevant_context": relevant_context,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "answer": "I apologize, but I encountered an error while processing your question. Please try rephrasing it.",
            "relevant_context": "",
            "status": "error",
            "error": str(e)
        }
