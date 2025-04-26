import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Tuple, List
import io
from PIL import Image
import base64
import re

def extract_sections_from_pdf(pdf_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Extract both text sections and images from a PDF file.
    Returns a tuple of (sections, images) where:
    - sections is a dictionary of section_name -> section_text
    - images is a list of dictionaries containing image data and metadata
    """
    doc = fitz.open(pdf_path)
    sections = {}
    images = []
    current_section = None
    current_text = []
    
    # Main section headers only - using re.IGNORECASE flag instead of inline (?i)
    section_patterns = {
        r"^(?:\d+[\.\s]*)?\s*(?:abstract|summary)\s*:?\s*$": "Abstract",
        r"^(?:\d+[\.\s]*)?\s*(?:introduction|background)\s*:?\s*$": "Introduction",
        r"^(?:\d+[\.\s]*)?\s*(?:methods?|methodology|materials\s+and\s+methods)\s*:?\s*$": "Methods",
        r"^(?:\d+[\.\s]*)?\s*(?:results?|findings)\s*:?\s*$": "Results",
        r"^(?:\d+[\.\s]*)?\s*(?:discussion|analysis)\s*:?\s*$": "Discussion",
        r"^(?:\d+[\.\s]*)?\s*(?:conclusions?|concluding\s+remarks)\s*:?\s*$": "Conclusion",
        r"^(?:\d+[\.\s]*)?\s*(?:references|bibliography)\s*:?\s*$": "References"
    }

    # Additional patterns that might indicate a subsection (to be ignored)
    subsection_patterns = [
        r"^\d+\.\d+",  # Numbered subsections like "1.1", "2.3"
        r"^[A-Z]\.",   # Letter subsections like "A.", "B."
        r"^figure",    # Figures
        r"^table",     # Tables
        r"^appendix",  # Appendices
        r"^acknowledgments?",  # Acknowledgments
        r"^supplementary"  # Supplementary material
    ]
    
    def is_subsection(text: str) -> bool:
        """Check if text appears to be a subsection header."""
        text = text.strip()
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in subsection_patterns)

    def is_section_header(text: str, font_size: float, is_bold: bool) -> str:
        """Check if text is a main section header and return standardized name if it is."""
        text = text.strip()
        
        # Ignore subsections
        if is_subsection(text):
            return None
            
        # Check against main section patterns
        for pattern, section_name in section_patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                return section_name
                
        # Only consider custom sections if they're clearly emphasized
        if (len(text) < 50 and  # Reasonably short
            text.isupper() and  # ALL CAPS
            is_bold and  # Bold text
            font_size > avg_font_size * 1.2):  # Significantly larger font
            return text.title()  # Convert "NEURAL NETWORKS" to "Neural Networks"
            
        return None

    # First pass: determine average font size
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > 0:  # Ignore zero font sizes
                            font_sizes.append(span["size"])
    
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
    
    # Second pass: extract content
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        # Check for section header
                        is_bold = bool(span.get("flags", 0) & 2**1)  # Check if text is bold
                        section_name = is_section_header(text, span["size"], is_bold)
                        
                        if section_name:
                            # Save previous section
                            if current_section and current_text:
                                sections[current_section] = "\n".join(current_text).strip()
                            current_section = section_name
                            current_text = []
                        elif current_section:  # Only collect text if we're in a section
                            line_text.append(text)
                    
                    if line_text and current_section:
                        current_text.append(" ".join(line_text))
        
        # Add a newline between pages for better readability
        if current_text:
            current_text.append("\n")
        
        # Extract images (unchanged)
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get image metadata
            width, height = image.size
            format = image.format
            
            # Convert to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Store image data with metadata
            images.append({
                "page": page_num,
                "index": img_index,
                "width": width,
                "height": height,
                "format": format,
                "data": img_str,
                "caption": f"Figure {len(images) + 1} from page {page_num + 1}"
            })
    
    # Save the last section
    if current_section and current_text:
        sections[current_section] = "\n".join(current_text).strip()
    
    # Clean up sections
    cleaned_sections = {}
    for name, text in sections.items():
        # Remove multiple newlines and clean up spacing
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', text)
        cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text)
        if cleaned_text:  # Only keep non-empty sections
            cleaned_sections[name] = cleaned_text
    
    # If no main sections were found, create a single section with all text
    if not cleaned_sections:
        all_text = []
        for page in doc:
            all_text.append(page.get_text())
        cleaned_sections["Content"] = "\n".join(all_text)
    
    return cleaned_sections, images
