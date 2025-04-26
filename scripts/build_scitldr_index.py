import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import ResearchPaperRAG
import pandas as pd
from pathlib import Path

def build_scitldr_index(csv_path: str, index_dir: str = "index"):
    """Build a search index from the scitldr dataset."""
    # Initialize RAG system
    rag = ResearchPaperRAG(index_dir)
    
    # Load the dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Add each paper's abstract and TLDR to the index
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing entry {idx}...")
            
        # Add the abstract with its metadata
        metadata = {
            'type': 'abstract',
            'paper_id': f'scitldr_{idx}',
            'title': row.get('title', 'N/A'),
            'has_tldr': True
        }
        rag.add_to_index(row['abstract'], metadata)
        
        # Add the TLDR with its metadata
        metadata = {
            'type': 'tldr',
            'paper_id': f'scitldr_{idx}',
            'title': row.get('title', 'N/A'),
            'is_tldr': True
        }
        rag.add_to_index(row['tldr'], metadata)
    
    print(f"\nIndex built successfully! Stats:")
    print(f"Total documents indexed: {len(rag.stored_data)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a search index from scitldr dataset")
    parser.add_argument("--csv-path", default="data/scitldr_clean.csv", help="Path to scitldr_clean.csv")
    parser.add_argument("--index-dir", default="index", help="Directory to store the index")
    
    args = parser.parse_args()
    build_scitldr_index(args.csv_path, args.index_dir) 