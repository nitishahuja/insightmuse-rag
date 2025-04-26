import os
from PIL import Image, ImageStat
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import io

class ImageAnalyzer:
    def __init__(self):
        pass

    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[str, Image.Image]]:
        """Extract images from a PDF file."""
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append((f"page_{page_num}_img_{img_index}", image))
        
        return images

    def analyze_image(self, image: Image.Image) -> str:
        """Analyze an image using basic image processing techniques."""
        # Get image dimensions
        width, height = image.size
        
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Calculate basic statistics
        stats = ImageStat.Stat(gray)
        mean_brightness = stats.mean[0]
        std_brightness = stats.stddev[0]
        
        # Convert to numpy array for edge detection approximation
        img_array = np.array(gray)
        gradient_x = np.gradient(img_array, axis=1)
        gradient_y = np.gradient(img_array, axis=0)
        edge_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_density = np.mean(edge_magnitude > 30)  # threshold of 30
        
        # Determine if image is likely a plot/graph based on edge density
        is_plot = edge_density > 0.1
        
        # Generate basic description
        description = f"""Image Analysis:
- Dimensions: {width}x{height} pixels
- Average brightness: {mean_brightness:.1f}/255
- Brightness variation: {std_brightness:.1f}
- Edge density: {edge_density:.3f}
- Type: {"Plot/Graph" if is_plot else "Photo/Illustration"}
"""
        return description

    def process_pdf_images(self, pdf_path: str) -> Dict[str, str]:
        """Process all images in a PDF and return their analyses."""
        images = self.extract_images_from_pdf(pdf_path)
        analyses = {}
        
        for img_name, img in images:
            analysis = self.analyze_image(img)
            analyses[img_name] = analysis
            
        return analyses

    def save_image_analysis(self, analyses: Dict[str, str], output_dir: str = "outputs"):
        """Save image analyses to a text file."""
        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / "image_analyses.txt"
        
        with open(output_path, "w") as f:
            for img_name, analysis in analyses.items():
                f.write(f"=== {img_name} ===\n")
                f.write(f"{analysis}\n\n")
        
        return str(output_path) 