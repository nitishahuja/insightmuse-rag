"""
Prompt templates for text simplification and information extraction.
These prompts guide the behavior of language models for structured tasks.
"""

def simplify_prompt(text: str) -> str:
    """
    Generate a prompt for simplifying complex research text.

    Args:
        text (str): The input abstract or section text.

    Returns:
        str: Formatted prompt for simplification.
    """
    return f"""Simplify and explain the following text in clear, simple language. Include key points as bullet points:

{text}

Simplified explanation:"""


def multimodal_simplify_prompt(text: str, image_description: str) -> str:
    """
    Generate a prompt for simplification using both text and accompanying image description.

    Args:
        text (str): Main section text.
        image_description (str): Caption or description of the related figure/image.

    Returns:
        str: Formatted multimodal prompt.
    """
    return f"""Simplify and explain the following text and its associated figure in clear, simple language. Include key points as bullet points:

Text:
{text}

Figure description:
{image_description}

Simplified explanation:"""


def extract_steps_prompt(text: str) -> str:
    """
    Prompt to extract numbered, sequential steps from a block of scientific text.

    Args:
        text (str): Typically a methods or procedure section.

    Returns:
        str: Prompt for extracting steps.
    """
    return f"""Extract and number the key steps or procedures described in this text. Format as a clear, sequential list:

{text}

Steps:"""


def extract_table_prompt(text: str) -> str:
    """
    Prompt for converting text into tabular form with parameters, metrics, and values.

    Args:
        text (str): Usually from results or metrics sections.

    Returns:
        str: Prompt for table extraction.
    """
    return f"""Extract key parameters, values, and relationships from this text into a structured format. Include units where applicable:

{text}

Structured information:"""
