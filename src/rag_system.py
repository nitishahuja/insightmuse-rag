import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import logging

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

class ResearchPaperRAG:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.papers = []
        self.stored_data = []  # Store all indexed documents
        self.load_index()

    def load_index(self):
        """Load or create the FAISS index."""
        try:
            if os.path.exists(f"{self.index_path}.index"):
                self.index = faiss.read_index(f"{self.index_path}.index")
                if os.path.exists(f"{self.index_path}.json"):
                    try:
                        with open(f"{self.index_path}.json", 'r') as f:
                            self.papers = json.load(f)
                            self.stored_data = self.papers
                    except json.JSONDecodeError:
                        logging.warning("Corrupted JSON file detected. Creating new index.")
                        self._create_new_index()
                else:
                    self._create_new_index()
            else:
                self._create_new_index()
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty index."""
        self.index = faiss.IndexFlatL2(384)  # Dimension of all-MiniLM-L6-v2 embeddings
        self.papers = []
        self.stored_data = []
        self.save_index()  # Save the empty index

    def save_index(self):
        """Save the FAISS index and paper metadata."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save papers metadata with proper JSON formatting
            with open(f"{self.index_path}.json", 'w') as f:
                json.dump(self.papers, f, indent=2)  # Use indent for better readability
                
            logging.info(f"Index saved successfully with {len(self.papers)} documents")
        except Exception as e:
            logging.error(f"Error saving index: {str(e)}")
            raise

    def add_to_index(self, text: str, metadata: Dict):
        """Add a document to the index with metadata."""
        # Generate embedding for the text
        embedding = self.model.encode([text])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store document with metadata
        doc = {
            'text': text,
            'metadata': metadata
        }
        self.papers.append(doc)
        self.stored_data.append(doc)
        
        # Save the updated index
        self.save_index()

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if not self.papers:
            logging.warning("No documents in the index. Returning empty list.")
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Return the most similar documents
        try:
            return [self.papers[idx] for idx in indices[0]]
        except IndexError:
            logging.error("Error accessing documents from index. Returning empty list.")
            return []

    def generate_tldr(self, text: str) -> str:
        """Generate a TLDR summary of the text using the pre-trained knowledge."""
        # First, find similar abstracts from scitldr
        similar_docs = self.search_similar(text, k=3)
        
        # Use the similar abstracts to help generate a better TLDR
        context = ""
        for doc in similar_docs:
            if doc['metadata'].get('type') == 'tldr':
                context += f"Example TLDR: {doc['text']}\n\n"
        
        prompt = f"""
        Based on these example TLDRs from similar research papers:
        {context}
        
        Please provide a concise TLDR summary of the following research paper section:
        
        {text}
        
        Focus on the key points and main findings. Keep it brief but informative.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that provides clear and concise summaries of academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()

    def answer_question(self, question: str, context: Dict[str, str]) -> str:
        """Answer a question about the research paper."""
        # First, find relevant sections from the uploaded paper
        similar_docs = self.search_similar(question, k=3)
        relevant_sections = []
        for doc in similar_docs:
            if doc['metadata'].get('type') == 'section':
                relevant_sections.append({
                    'title': doc['metadata'].get('title', 'Section'),
                    'text': doc['text']
                })
        
        # Combine all sections into a single context, but limit the total length
        max_context_length = 12000  # Leave some room for the question and system message
        current_length = 0
        full_context = []
        
        for section, text in context.items():
            # If adding this section would exceed the limit, truncate it
            if current_length + len(text) > max_context_length:
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Only add if we have enough space for meaningful content
                    full_context.append(f"{section}:\n{text[:remaining_length]}...")
                break
            full_context.append(f"{section}:\n{text}")
            current_length += len(text)
        
        full_context = "\n\n".join(full_context)
        
        # Prepare references from relevant sections
        references = ""
        if relevant_sections:
            references = "\n\n### References from the Paper:\n"
            for section in relevant_sections:
                references += f"\n**{section['title']}**:\n{section['text'][:500]}...\n"
        
        prompt = f"""
        Context from the research paper:
        {full_context}
        
        Question: {question}
        
        Please provide a detailed answer based on the context. Include specific references to the paper where relevant.
        If the answer cannot be found in the context, say so.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant that answers questions about academic papers. Always cite specific sections or findings from the paper when possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Combine the answer with the references
            answer = response.choices[0].message.content.strip()
            if relevant_sections:
                answer += references
            
            return answer
        except Exception as e:
            logging.error(f"Error in answer_question: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it or ask about a different aspect of the paper."

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Get relevant context from the uploaded paper."""
        # Search through the stored data (which includes the uploaded paper's sections)
        similar_docs = self.search_similar(query, k)
        
        if not similar_docs:
            return "No relevant sections found in the paper."
        
        context = "### Relevant Sections from the Paper:\n\n"
        for doc in similar_docs:
            if doc['metadata'].get('type') == 'section':
                context += f"**{doc['metadata'].get('title', 'Section')}**:\n{doc['text']}\n\n"
        
        return context 