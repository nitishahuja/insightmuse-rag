import sys
sys.path.append(".")

from src.visualizer import extract_steps_from_abstract, generate_flowchart

if __name__ == "__main__":
    abstract = """Denoising diffusion probabilistic models (DDPMs) have recently emerged as a powerful class of generative models... (insert the full abstract here)"""

    steps = extract_steps_from_abstract(abstract)
    print("\n🧩 Extracted Steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

    generate_flowchart(steps)
