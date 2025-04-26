import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import ResearchPaperRAG
from src.pdf_extractor import extract_sections_from_pdf
from pathlib import Path
import json

def process_paper(pdf_path: str, rag: ResearchPaperRAG) -> None:
    """Process a single paper and add its sections to the index."""
    print(f"Processing {pdf_path}...")
    
    # Extract sections and images
    sections, images = extract_sections_from_pdf(pdf_path)

    # Add each section to the index
    for section_name, section_text in sections.items():
        metadata = {
            'paper_title': Path(pdf_path).stem,
            'section_name': section_name,
            'source_file': pdf_path
        }
        rag.add_to_index(section_text, metadata)
        print(f"Added section: {section_name}")

def build_index(papers_dir: str, index_dir: str = "index"):
    """Build a search index from a directory of research papers."""
    # Initialize RAG system
    rag = ResearchPaperRAG(index_dir)
    
    # Process each PDF in the directory
    papers_path = Path(papers_dir)
    for pdf_file in papers_path.glob("*.pdf"):
        process_paper(str(pdf_file), rag)
    
    print(f"\nIndex built successfully! Stats:")
    print(f"Total documents indexed: {len(rag.stored_data)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a search index from research papers")
    parser.add_argument("papers_dir", help="Directory containing PDF papers to index")
    parser.add_argument("--index-dir", default="index", help="Directory to store the index")
    
    args = parser.parse_args()
    build_index(args.papers_dir, args.index_dir)
