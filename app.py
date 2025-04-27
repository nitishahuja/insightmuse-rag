import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from src.rag_pipeline import simplify_with_prompt, simplify_section
from src.visualizer import visualize_section
from src.pdf_extractor import extract_sections_from_pdf
from src.rag_system import ResearchPaperRAG
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="InsightMuse", layout="wide")
st.title("🧠 InsightMuse — Research Paper Visualizer")

# Initialize session state
if 'sections' not in st.session_state:
    st.session_state.sections = None

st.sidebar.title("📁 Upload Input")
input_mode = st.sidebar.radio("Choose input type:", ["📄 Upload PDF", "📊 Upload CSV"])

# Initialize RAG system
@st.cache_resource
def init_rag():
    return ResearchPaperRAG("index")

rag = init_rag()

def determine_section_type(title: str) -> str:
    """Determine the type of section based on its title."""
    title_lower = title.lower()
    if any(word in title_lower for word in ["method", "procedure", "approach", "implementation"]):
        return "methodology"
    elif any(word in title_lower for word in ["result", "finding", "evaluation", "analysis"]):
        return "results"
    elif any(word in title_lower for word in ["discussion", "conclusion", "summary"]):
        return "discussion"
    return "other"

def display_visualization(viz_path: str, section_type: str):
    """Display visualization with appropriate context."""
    if viz_path and os.path.exists(viz_path):
        image_bytes = Path(viz_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">', unsafe_allow_html=True)
        
        # Add explanation based on section type
        if section_type == "methodology":
            st.info("📊 This flowchart visualizes the key steps and procedures in the methodology.")
        elif section_type == "results":
            st.info("📈 This chart presents the key findings and results statistically.")
        elif section_type == "discussion":
            st.info("🔄 This concept map shows the relationships between key ideas in the discussion.")
        
        # Return the base64 encoded image for API response
        return f"data:image/png;base64,{b64}"
    else:
        st.warning("⚠️ Could not generate visualization for this section.")
        return None

# --- PDF Mode ---
if input_mode == "📄 Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        st.success("✅ PDF uploaded. Processing sections...")
        sections, _ = extract_sections_from_pdf(temp_path)  # Ignore images
        os.unlink(temp_path)  # Clean up temp file
        st.session_state.sections = sections

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Visual Analysis", "❓ Q&A"])

        with tab1:
            st.header("Visual Paper Analysis")
            
            # Process each section
            processed_sections = []
            for section_name, section_text in sections.items():
                section_type = determine_section_type(section_name)
                with st.expander(f"{section_name} ({section_type})"):
                    # Generate TLDR
                    tldr = rag.generate_tldr(section_text)
                    st.markdown("### Key Points")
                    st.markdown(tldr["tldr"])  # Display just the TLDR text
                    
                    # Generate and display visualization
                    st.markdown("### Visual Representation")
                    visualization_data = None
                    with st.spinner("Generating visualization..."):
                        try:
                            # Generate unique visualization path for each section
                            viz_path = visualize_section(section_name, section_text)
                            visualization_data = display_visualization(viz_path, section_type)
                        except Exception as e:
                            st.error(f"❌ Error generating visualization: {str(e)}")
                    
                    # Store processed section data
                    processed_sections.append({
                        "title": section_name,
                        "section_type": section_type,
                        "tldr": tldr,
                        "visualization": visualization_data,
                        "status": "completed" if visualization_data else "error"
                    })
            
            # Store processed sections in session state for API access
            st.session_state.processed_sections = processed_sections

        with tab2:
            st.header("Ask Questions")
            question = st.text_input("Ask a question about the paper:")
            
            if question:
                answer = rag.answer_question(question, sections)
                st.markdown("### Answer")
                st.markdown(answer)
                
                with st.expander("View Related Research"):
                    st.markdown(rag.get_relevant_context(question, k=3))
    else:
        st.info("Please upload a PDF to begin analysis.")
        
        # Show example capabilities
        st.markdown("""
        ### 🎯 What This Tool Does:
        - 📊 Converts methodology sections into flowcharts
        - 📈 Transforms results into statistical visualizations
        - 🔄 Creates concept maps for discussions and conclusions
        - 📝 Generates concise summaries for each section
        - ❓ Answers questions about the paper
        """)

# --- CSV Mode ---
elif input_mode == "📊 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with an 'abstract' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "abstract" not in df.columns:
            st.error("CSV must contain an 'abstract' column.")
        else:
            st.write("### Abstract Analysis:")
            for idx, abstract in enumerate(df['abstract']):
                with st.expander(f"Abstract {idx + 1}"):
                    # Generate TLDR
                    tldr = rag.generate_tldr(abstract)
                    st.markdown("### Key Points")
                    st.markdown(tldr["tldr"])  # Display just the TLDR text
                    
                    # Generate visualization
                    st.markdown("### Visual Representation")
                    with st.spinner("Generating visualization..."):
                        try:
                            viz_path = visualize_section(f"Abstract {idx + 1}", abstract)
                            display_visualization(viz_path, "other")
                        except Exception as e:
                            st.error(f"❌ Error generating visualization: {str(e)}")
