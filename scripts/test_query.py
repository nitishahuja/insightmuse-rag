import sys
sys.path.append(".")

from src.rag_pipeline import search_top_k, simplify_with_prompt

if __name__ == "__main__":
    query = "critical points in neural networks"
    print(f"\n🔍 Query: {query}\n")

    top_papers = search_top_k(query, k=3)
    
    for i, paper in enumerate(top_papers):
        title = paper['title'] if isinstance(paper['title'], str) and paper['title'] else "[No Title]"
        print(f"\n📄 Title {i+1}: {title}")

        print(f"🔬 Original Abstract:\n{paper['abstract'][:300]}...\n")
        
        simplified = simplify_with_prompt(paper["abstract"])
        print(f"✅ Simplified Summary:\n{simplified}\n")
