import pandas as pd
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

def build_faiss_index(csv_path="data/scitldr_clean.csv", index_path="index/abstract.index", metadata_path="index/metadata.pkl"):
    # Load dataset
    df = pd.read_csv(csv_path)
    abstracts = df["abstract"].tolist()

    # Load embedding model
    print("🔍 Embedding abstracts...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(abstracts, show_progress_bar=True)

    # Build FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index built with {len(embeddings)} entries")

    # Save index
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"💾 Saved index to {index_path}")

    # Save metadata (for retrieval later)
    metadata = df[["id", "title", "abstract", "tldr"]].to_dict(orient="records")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"💾 Saved metadata to {metadata_path}")
