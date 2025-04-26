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
st.title("🧠 InsightMuse — Research Paper Explainer")

# Initialize session state
if 'sections' not in st.session_state:
    st.session_state.sections = None
if 'simplified_sections' not in st.session_state:
    st.session_state.simplified_sections = None

st.sidebar.title("📁 Upload Input")
input_mode = st.sidebar.radio("Choose input type:", ["📄 Upload PDF", "📊 Upload CSV"])

# Initialize RAG system
@st.cache_resource
def init_rag():
    return ResearchPaperRAG("index")

rag = init_rag()

# --- PDF Mode ---
if input_mode == "📄 Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        st.success("✅ PDF uploaded. Extracting sections...")
        sections, _ = extract_sections_from_pdf(temp_path)  # Ignore images
        os.unlink(temp_path)  # Clean up temp file
        st.session_state.sections = sections

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📝 Paper Analysis", "❓ Q&A"])

        with tab1:
            st.header("Paper Analysis")
            
            # Process each section
            for section_name, section_text in sections.items():
                with st.expander(f"{section_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Text")
                        st.text_area("", section_text, height=300, key=f"orig_{section_name}")
                    
                    with col2:
                        st.subheader("TLDR")
                        # Generate TLDR using examples from scitldr dataset
                        tldr = rag.generate_tldr(section_text)
                        st.markdown(tldr)

        with tab2:
            st.header("Ask Questions")
            question = st.text_input("Ask a question about the paper:")
            
            if question:
                answer = rag.answer_question(question, sections)
                st.markdown("### Answer")
                st.markdown(answer)
                
                with st.expander("View Related Research"):
                    st.markdown(rag.get_relevant_context(question, k=5))
    else:
        st.info("Please upload a PDF to begin analysis.")
        
        # Show some example questions that can be asked
        st.markdown("""
        ### Example Questions You Can Ask:
        - What are the main contributions of this paper?
        - How does this research compare to similar work in the field?
        - What methods were used in this study?
        - What are the key findings and their implications?
        - Can you explain the methodology in simpler terms?
        """)

# --- CSV Mode ---
elif input_mode == "📊 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with an 'abstract' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "abstract" not in df.columns:
            st.error("CSV must contain an 'abstract' column.")
        else:
            st.write("### Abstracts in the CSV:")
            for idx, abstract in enumerate(df['abstract']):
                with st.expander(f"Abstract {idx + 1}"):
                    st.write(abstract)
                    with st.spinner("Generating visualization..."):
                        try:
                            viz_path = visualize_section(f"Abstract {idx + 1}", abstract)
                            if viz_path and os.path.exists(viz_path):
                                image_bytes = Path(viz_path).read_bytes()
                                b64 = base64.b64encode(image_bytes).decode()
                                st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">', unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ Could not generate visualization for this abstract. The content might be too complex or abstract.")
                        except Exception as e:
                            st.error(f"❌ Error generating visualization: {str(e)}")
