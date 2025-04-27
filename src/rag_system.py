import os
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchPaperRAG:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.papers = []  # Store uploaded papers
        self.stored_data = []  # Store text and metadata
        
        # Initialize or load FAISS index
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of the embeddings
        
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

    def add_to_index(self, text: str, metadata: Dict = None) -> None:
        """Add a piece of text and its metadata to the index."""
        try:
            # Get embedding
            embedding = self.encoder.encode([text])[0]
        
            # Add to index
            self.index.add(np.array([embedding]).astype('float32'))
        
            # Store text and metadata
            self.stored_data.append({
            'text': text,
                'metadata': metadata or {}
            })
            
        except Exception as e:
            logger.error(f"Error adding to index: {str(e)}")

    def search_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        try:
            # Get query embedding
            query_embedding = self.encoder.encode([query])[0]
        
            # Search index
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
            # Get results
            results = []
            for idx in I[0]:
                if idx < len(self.stored_data):  # Ensure index is valid
                    results.append(self.stored_data[idx])
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []

    def generate_tldr(self, text: str) -> Dict:
        """Generate a TLDR summary of the text using the pre-trained knowledge and suggest visualizations."""
        # First, find similar abstracts from scitldr
        similar_docs = self.search_similar(text, k=3)
        
        # Use the similar abstracts to help generate a better TLDR
        context = ""
        for doc in similar_docs:
            if doc['metadata'].get('type') == 'tldr':
                context += f"Example TLDR: {doc['text']}\n\n"
        
        prompt = f"""Based on these example TLDRs from similar research papers:
{context}

Please provide a concise TLDR summary and visualization suggestion in the following format:

{{
    "tldr": "your concise summary here",
    "visualization": {{
        "viz_type": "NONE",  // Default choice, only change if visualization truly adds value
        "explanation": "why visualization is not needed or how it would help",
        "data": {{}}  // Only include if suggesting a visualization
    }}
}}

Text to analyze:
{text}

IMPORTANT:
- Keep the summary clear and concise
- Default to NONE for visualization unless it truly adds value
- Ensure the response is valid JSON
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research assistant that provides clear summaries and ONLY suggests visualizations when they truly add value. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            try:
                result = json.loads(response.choices[0].message.content.strip())
                return {
                    "tldr": result.get("tldr", "Error parsing summary"),
                    "visualization": result.get("visualization", {
                        "viz_type": "NONE",
                        "explanation": "Error parsing visualization data"
                    })
                }
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Raw response: {response.choices[0].message.content}")
                return {
                    "tldr": response.choices[0].message.content.strip(),
                    "visualization": {
                        "viz_type": "NONE",
                        "explanation": "Error parsing response as JSON"
                    }
                }
            
        except Exception as e:
            logging.error(f"Error in generate_tldr: {str(e)}")
            return {
                "tldr": "Error generating summary",
                "visualization": {
                    "viz_type": "NONE",
                    "explanation": f"Error in analysis: {str(e)}"
                }
            }

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