import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env file in project root

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load everything once
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_INDEX = faiss.read_index("index/abstract.index")
with open("index/metadata.pkl", "rb") as f:
    METADATA = pickle.load(f)


def embed_query(query: str):
    return EMBEDDING_MODEL.encode([query])[0]


def search_top_k(query: str, k=5) -> List[dict]:
    query_vec = embed_query(query)
    distances, indices = FAISS_INDEX.search(query_vec.reshape(1, -1), k)
    results = [METADATA[i] for i in indices[0]]
    return results


def simplify_with_prompt(abstract: str, model="gpt-3.5-turbo") -> str:
    prompt = f"""
You are a research assistant who simplifies complex research abstracts for graduate students.
Explain the following abstract in very simple, clear language with key takeaways in bullet points.

Abstract:
\"\"\"{abstract}\"\"\"
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content


# 🧠 NEW: Simplify a specific section of a paper (e.g., Methods, Results, etc.)
def simplify_section(title: str, content: str, model="gpt-3.5-turbo") -> str:
    prompt = f"""
You are a helpful research assistant. Simplify the following research paper section titled '{title}' in clear and accessible language. Use concise points, short paragraphs, and highlight the core purpose or steps taken in that section.

Section Title: {title}

Content:
\"\"\"{content}\"\"\"
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content
