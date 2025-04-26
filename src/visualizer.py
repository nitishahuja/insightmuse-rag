import graphviz
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import os
from openai import OpenAI
from dotenv import load_dotenv
import seaborn as sns
import json
import html
import logging
import sys
import traceback
import networkx as nx
from pathlib import Path

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def analyze_content_for_visualization(title: str, content: str) -> Dict:
    """Analyze the content to determine the best visualization type and extract relevant data."""
    prompt = f"""
Analyze the following research paper section and create a detailed visualization plan:

Title: {title}
Content: {content}

First, analyze the content to understand:
1. The main concepts and their relationships
2. Any numerical data or comparisons
3. The flow of information or process steps
4. Key findings or results
5. Important patterns or trends

Then, create a comprehensive visualization plan that includes:
1. The most appropriate visualization type
2. Detailed data extraction
3. Clear labeling and annotations
4. Meaningful color schemes
5. Proper scaling and layout

Return a JSON object with the following structure:
{{
    "visualization_type": "flowchart" | "bar_chart" | "line_chart" | "pie_chart" | "process_diagram" | 
                         "scatter_plot" | "box_plot" | "heatmap" | "concept_map" | "hierarchical_tree" | 
                         "venn_diagram" | "radar_chart",
    "title": "descriptive title that captures the main insight",
    "data": {{
        "labels": ["label1", "label2"],
        "values": [1, 2],
        "annotations": ["insight1", "insight2"],
        "units": "units",
        "baseline": 0,
        "highlight_slices": [0, 1],
        "total": 100,
        "steps": ["step1", "step2"],
        "decision_points": ["decision1", "decision2"],
        "branches": [["condition1", "outcome1"]],
        "feedback_loops": [["step1", "step2"]],
        "x_values": [1, 2],
        "y_values": [1, 2],
        "clusters": [{{"points": [0, 1], "label": "cluster1"}}],
        "trend_lines": [{{"type": "linear", "degree": 1}}],
        "outliers": [{{"group": 0, "value": 10}}],
        "statistics": ["mean", "median"],
        "highlight_cells": [{{"row": 0, "col": 0}}],
        "color_range": {{"min": 0, "max": 100}},
        "nodes": ["node1", "node2"],
        "connections": [["node1", "node2", "relation"]],
        "hierarchies": [["parent", "child"]],
        "key_concepts": ["concept1", "concept2"],
        "levels": [["root"], ["level1"]],
        "leaf_nodes": ["leaf1", "leaf2"],
        "sets": [{{"name": "set1", "elements": ["elem1"]}}],
        "intersections": [{{"sets": ["set1", "set2"], "elements": ["elem1"]}}],
        "unique_elements": [{{"set": "set1", "elements": ["elem1"]}}],
        "ranges": [{{"min": 0, "max": 100}}]
    }},
    "description": "detailed explanation of what the visualization shows and its significance",
    "style": {{
        "color_scheme": "academic",
        "font_size": 20,
        "show_legend": true,
        "show_grid": true,
        "annotations": {{
            "show_values": true,
            "show_percentages": true,
            "show_trends": true,
            "highlight_key_points": true
        }},
        "layout": {{
            "orientation": "vertical",
            "spacing": 1.0,
            "margins": {{"top": 0.1, "right": 0.1, "bottom": 0.1, "left": 0.1}},
            "aspect_ratio": 1.5
        }}
    }}
}}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant that analyzes academic content and suggests appropriate visualizations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # If JSON parsing fails, return a default flowchart visualization
        return {
            "visualization_type": "flowchart",
            "title": title,
            "data": {
                "steps": [content],
                "decision_points": [],
                "branches": [],
                "feedback_loops": []
            },
            "description": "Default flowchart visualization",
            "style": {
                "color_scheme": "academic",
                "font_size": 20,
                "show_legend": True,
                "show_grid": True,
                "annotations": {
                    "show_values": True,
                    "show_percentages": True,
                    "show_trends": True,
                    "highlight_key_points": True
                },
                "layout": {
                    "orientation": "vertical",
                    "spacing": 1.0,
                    "margins": {"top": 0.1, "right": 0.1, "bottom": 0.1, "left": 0.1},
                    "aspect_ratio": 1.5
                }
            }
        }

def generate_flowchart(steps: List[str], title: str, style: Dict = None) -> str:
    """Generate a flowchart using matplotlib."""
    try:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Calculate positions for boxes
        num_steps = len(steps)
        box_height = 0.8
        vertical_spacing = 1.5
        total_height = num_steps * vertical_spacing
        
        # Draw boxes and arrows
        for i, step in enumerate(steps):
            # Calculate position
            y_pos = total_height - (i * vertical_spacing)
            
            # Draw box
            rect = plt.Rectangle((0.2, y_pos - box_height/2), 0.6, box_height,
                               facecolor='#f8f9fa',
                               edgecolor='#2c3e50',
                               alpha=0.8,
                               linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add text
            step_text = step[:100] + '...' if len(step) > 100 else step
            plt.text(0.5, y_pos, step_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    wrap=True,
                    fontsize=10)
            
            # Draw arrow
            if i < num_steps - 1:
                plt.arrow(0.5, y_pos - box_height/2 - 0.1,
                         0, -vertical_spacing + box_height + 0.2,
                         head_width=0.05,
                         head_length=0.1,
                         fc='#2c3e50',
                         ec='#2c3e50')
        
        # Set title
        plt.title(title, pad=20, fontsize=14, wrap=True)
        
        # Set layout
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join("outputs", "visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        logging.error(f"Error generating flowchart: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def generate_error_visualization(error_message: str) -> str:
    """Generate an error visualization using matplotlib."""
    try:
        logging.error(f"Generating error visualization for: {error_message}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating visualization:\n{error_message}",
                ha='center', va='center',
                wrap=True,
                fontsize=12,
                color='red')
        plt.axis('off')
        error_path = os.path.join("outputs", "visualization_error.png")
        plt.savefig(error_path, bbox_inches='tight')
        plt.close()
        return error_path
    except Exception as e:
        logging.critical(f"Failed to generate error visualization: {str(e)}")
        logging.critical(traceback.format_exc())
        return ""

def generate_scatter_plot(x_values: List[float], y_values: List[float], labels: List[str], title: str, style: Dict = None) -> str:
    """Generate a scatter plot with optional labels."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_values, y_values, c='#3498db', alpha=0.6)
    
    # Add labels if provided
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]))
    
    plt.title(title)
    if style.get('show_grid', True):
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.close()
    return "outputs/visualization.png"

def generate_box_plot(data: List[List[float]], labels: List[str], title: str, style: Dict = None) -> str:
    """Generate a box plot for comparing distributions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.title(title)
    if style.get('show_grid', True):
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.close()
    return "outputs/visualization.png"

def generate_heatmap(data: List[List[float]], row_labels: List[str], col_labels: List[str], title: str, style: Dict = None) -> str:
    """Generate a heatmap for correlation or matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.close()
    return "outputs/visualization.png"

def generate_concept_map(nodes: List[str], connections: List[Tuple[str, str, str]], title: str, style: Dict = None) -> str:
    """Generate a concept map showing relationships between concepts."""
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    
    # Add nodes
    for node in nodes:
        dot.node(node, shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Add connections
    for source, target, relation in connections:
        dot.edge(source, target, label=relation)
    
    dot.attr(label=title, labelloc='t', fontsize=str(style.get('font_size', 20)))
    output_path = dot.render(filename="outputs/visualization", cleanup=True)
    return output_path

def generate_venn_diagram(sets: List[Dict], title: str, style: Dict = None) -> str:
    """Generate a Venn diagram showing set relationships."""
    from matplotlib_venn import venn2, venn3
    
    plt.figure(figsize=(10, 8))
    
    if len(sets) == 2:
        venn2(subsets=(len(sets[0]['elements']), len(sets[1]['elements']), 
                      len(set(sets[0]['elements']) & set(sets[1]['elements']))),
              set_labels=(sets[0]['name'], sets[1]['name']))
    elif len(sets) == 3:
        venn3(subsets=(len(sets[0]['elements']), len(sets[1]['elements']), len(sets[2]['elements']),
                      len(set(sets[0]['elements']) & set(sets[1]['elements'])),
                      len(set(sets[1]['elements']) & set(sets[2]['elements'])),
                      len(set(sets[0]['elements']) & set(sets[2]['elements'])),
                      len(set(sets[0]['elements']) & set(sets[1]['elements']) & set(sets[2]['elements']))),
              set_labels=(sets[0]['name'], sets[1]['name'], sets[2]['name']))
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.close()
    return "outputs/visualization.png"

def generate_radar_chart(categories: List[str], values: List[float], title: str, style: Dict = None) -> str:
    """Generate a radar chart for multi-dimensional comparison."""
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.close()
    return "outputs/visualization.png"

def analyze_content_type(text: str) -> str:
    """Analyze the content to determine the most appropriate visualization type."""
    try:
        logging.info("Analyzing content type...")
        # Count numbers and identify patterns
        numbers = len([word for word in text.split() if word.replace('.','').replace('%','').isdigit()])
        has_percentages = '%' in text
        has_comparisons = any(word in text.lower() for word in ['versus', 'vs', 'compared to', 'more than', 'less than'])
        has_steps = any(word in text.lower() for word in ['first', 'second', 'then', 'next', 'finally', 'step'])
        has_categories = any(word in text.lower() for word in ['types', 'categories', 'classes', 'kinds', 'groups'])
        has_time = any(word in text.lower() for word in ['year', 'month', 'time', 'period', 'duration'])
        
        logging.debug(f"Content analysis results: numbers={numbers}, percentages={has_percentages}, "
                     f"comparisons={has_comparisons}, steps={has_steps}, categories={has_categories}, "
                     f"time={has_time}")
        
        # Determine visualization type based on content analysis
        if has_steps:
            viz_type = "flowchart"
        elif has_time and numbers:
            viz_type = "timeline"
        elif has_percentages or (has_comparisons and numbers):
            viz_type = "bar_chart"
        elif has_categories:
            viz_type = "pie_chart"
        else:
            viz_type = "concept_map"
            
        logging.info(f"Selected visualization type: {viz_type}")
        return viz_type
    except Exception as e:
        logging.error(f"Error in analyze_content_type: {str(e)}")
        logging.error(traceback.format_exc())
        return "flowchart"  # Default to flowchart on error

def generate_bar_chart(data: List[str], title: str) -> str:
    """Generate a bar chart visualization."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Extract numbers from text
        numbers = []
        labels = []
        for item in data:
            # Find numbers in the text
            import re
            nums = re.findall(r'\d+(?:\.\d+)?', item)
            if nums:
                numbers.append(float(nums[0]))
                # Create label from text before the number
                label = re.split(r'\d+', item)[0].strip()
                labels.append(label[:20] + '...' if len(label) > 20 else label)
        
        if not numbers:
            return generate_error_visualization("No numerical data found for bar chart")
        
        # Create bar chart
        bars = plt.bar(range(len(numbers)), numbers)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:g}',
                    ha='center', va='bottom')
        
        plt.title(title)
        plt.tight_layout()
        
        output_path = os.path.join("outputs", "visualization.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        return output_path
    except Exception as e:
        return generate_error_visualization(str(e))

def generate_pie_chart(data: List[str], title: str) -> str:
    """Generate a pie chart visualization."""
    try:
        plt.figure(figsize=(10, 8))
        
        # Count category mentions
        from collections import Counter
        words = ' '.join(data).lower().split()
        categories = [word for word in words if len(word) > 3]  # Filter out short words
        category_counts = Counter(categories).most_common(5)  # Get top 5 categories
        
        if not category_counts:
            return generate_error_visualization("No categories found for pie chart")
        
        # Create pie chart
        labels = [cat[0] for cat in category_counts]
        sizes = [cat[1] for cat in category_counts]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
        
        output_path = os.path.join("outputs", "visualization.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        return output_path
    except Exception as e:
        return generate_error_visualization(str(e))

def generate_timeline(data: List[str], title: str) -> str:
    """Generate a timeline visualization."""
    try:
        plt.figure(figsize=(15, 8))
        
        # Extract time-related information
        import re
        time_events = []
        for item in data:
            # Look for years or dates
            years = re.findall(r'\b(19|20)\d{2}\b', item)
            if years:
                time_events.append((int(years[0]), item))
        
        if not time_events:
            return generate_error_visualization("No timeline data found")
        
        # Sort events by time
        time_events.sort(key=lambda x: x[0])
        
        # Create timeline
        levels = np.zeros(len(time_events))
        for i in range(len(time_events)):
            if i > 0:
                levels[i] = (levels[i-1] + 1) % 2
        
        # Plot events
        for i, (time, event) in enumerate(time_events):
            plt.plot([time, time], [0, levels[i]], 'k-', alpha=0.5)
            plt.plot(time, levels[i], 'ko', alpha=0.8)
            plt.annotate(event[:50] + '...' if len(event) > 50 else event,
                        xy=(time, levels[i]),
                        xytext=(10, 10 if levels[i] else -10),
                        textcoords='offset points',
                        ha='left',
                        va='bottom' if levels[i] else 'top')
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join("outputs", "visualization.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        return output_path
    except Exception as e:
        return generate_error_visualization(str(e))

def generate_concept_map(data: List[str], title: str) -> str:
    """Generate a concept map visualization."""
    try:
        logging.info("Generating concept map...")
        plt.figure(figsize=(12, 8))
        
        # Extract key concepts and relationships
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        
        # Create a graph
        G = nx.Graph()
        
        try:
            # Extract nouns as concepts
            all_text = ' '.join(data)
            tokens = word_tokenize(all_text)
            tagged = pos_tag(tokens)
            nouns = [word for word, pos in tagged if pos.startswith('NN')][:10]  # Get top 10 nouns
            
            logging.debug(f"Extracted nouns: {nouns}")
            
            if not nouns:
                return generate_error_visualization("No concepts found for concept map")
            
            # Add nodes and edges
            for i, noun in enumerate(nouns):
                G.add_node(noun)
                if i > 0:
                    G.add_edge(nouns[i-1], noun)
            
            # Draw the graph
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=2000, font_size=10, font_weight='bold')
            
            plt.title(title)
            plt.axis('off')
            
            output_path = os.path.join("outputs", "visualization.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info("Concept map generated successfully")
            return output_path
            
        except Exception as nlp_error:
            logging.error(f"Error in NLP processing: {str(nlp_error)}")
            logging.error(traceback.format_exc())
            return generate_error_visualization(f"Error in concept extraction: {str(nlp_error)}")
            
    except Exception as e:
        logging.error(f"Error generating concept map: {str(e)}")
        logging.error(traceback.format_exc())
        return generate_error_visualization(str(e))

def visualize_section(title: str, section_text: str) -> str:
    """
    Visualize a section based on content analysis.
    Returns the path to the generated visualization.
    """
    try:
        logging.info(f"Starting visualization for section: {title}")
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Input validation
        if not section_text or not title:
            error_msg = "Empty input: title or section_text is missing"
            logging.error(error_msg)
            return generate_error_visualization(error_msg)
        
        # Split text into manageable chunks
        chunks = [chunk.strip() for chunk in section_text.split('\n\n') if chunk.strip()]
        logging.debug(f"Number of text chunks: {len(chunks)}")
        
        # If the text is too long, use only the first few meaningful chunks
        if len(chunks) > 5:
            chunks = chunks[:5]
            logging.debug("Text truncated to first 5 chunks")
        
        # Analyze content and choose visualization type
        viz_type = analyze_content_type(section_text)
        logging.info(f"Visualization type selected: {viz_type}")
        
        # Generate appropriate visualization
        try:
            if viz_type == "flowchart":
                return generate_flowchart(chunks, title)
            elif viz_type == "bar_chart":
                return generate_bar_chart(chunks, title)
            elif viz_type == "pie_chart":
                return generate_pie_chart(chunks, title)
            elif viz_type == "timeline":
                return generate_timeline(chunks, title)
            else:
                return generate_concept_map(chunks, title)
        except Exception as viz_error:
            error_msg = f"Error generating {viz_type}: {str(viz_error)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return generate_error_visualization(error_msg)
            
    except Exception as e:
        error_msg = f"Error in visualize_section: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return generate_error_visualization(error_msg)
