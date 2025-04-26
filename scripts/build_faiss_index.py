import sys
sys.path.append(".")

from src.build_index import build_faiss_index

if __name__ == "__main__":
    build_faiss_index()
