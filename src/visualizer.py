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
import re

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def analyze_content_for_visualization(title: str, content: str) -> Dict:
    """Analyze content to determine the best visualization type and extract structured data."""
    try:
        # More strict content analysis
        technical_markers = ['algorithm', 'method', 'technique', 'procedure', 'implementation', 'system', 'framework']
        has_technical = any(marker in content.lower() for marker in technical_markers)
        
        step_markers = ['first', 'second', 'then', 'next', 'finally', 'step', 'process']
        has_steps = has_technical and any(marker in content.lower() for marker in step_markers)
        
        number_pattern = r'\d+\.?\d*\s*%|\d+\.?\d*'
        numbers = len(re.findall(number_pattern, content))
        comparison_terms = ['increased', 'decreased', 'higher than', 'lower than', 'compared to', 'versus', 'vs']
        has_comparisons = any(term in content.lower() for term in comparison_terms)
        
        logging.info(f"Content analysis: technical={has_technical}, steps={has_steps}, numbers={numbers}, comparisons={has_comparisons}")
        
        if not (has_steps or (numbers >= 3 and has_comparisons)):
            return {"viz_type": "NONE", "explanation": "Content is better understood through reading"}
        
        if has_technical and has_steps:
            steps = extract_steps(content)
            valid_steps = [s for s in steps if len(s.split()) >= 3 and len(s) >= 15]
            
            if len(valid_steps) >= 3:
                return {
                    "viz_type": "FLOWCHART",
                    "title": title,
                    "explanation": "Technical process flow visualization",
                    "data": {"steps": valid_steps},
                    "style": {"colors": ["#3498db", "#2ecc71"], "layout": "vertical"}
                }
        
        if numbers >= 3 and has_comparisons:
            data = extract_numerical_data(content)
            if len(data["values"]) >= 2 and all(v != 0 for v in data["values"]):
                return {
                    "viz_type": "BAR_CHART",
                    "title": title,
                    "explanation": "Numerical comparison visualization",
                    "data": data,
                    "style": {"colors": ["#3498db"], "layout": "vertical"}
                }
        
        return {"viz_type": "NONE", "explanation": "Content is better understood through reading"}
            
    except Exception as e:
        logging.error(f"Error in content analysis: {str(e)}")
        logging.error(traceback.format_exc())
        return {"viz_type": "NONE", "explanation": f"Error in analysis: {str(e)}"}

def extract_steps(content: str) -> List[str]:
    """Extract sequential steps from content, limiting to 3-4 words per step."""
    try:
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Look for numbered steps or step indicators
        steps = []
        step_indicators = ['first', 'second', 'third', 'next', 'then', 'finally', 'lastly']
        
        for sentence in sentences:
            # Check for numbered steps
            if any(f"{i}." in sentence or f"({i})" in sentence for i in range(1, 10)):
                steps.append(sentence)
            # Check for step indicators
            elif any(indicator in sentence.lower() for indicator in step_indicators):
                steps.append(sentence)
            # Check for bullet points or dashes
            elif sentence.strip().startswith(('•', '-', '*')):
                steps.append(sentence.strip('•- *'))
        
        # If no explicit steps found, try to break content into logical chunks
        if not steps and len(sentences) >= 3:
            steps = sentences[:5]  # Take up to 5 sentences
        
        # Clean and format steps - limit to 3-4 key words per step
        formatted_steps = []
        for step in steps[:5]:  # Limit to 5 steps maximum
            # Split into words and take key words
            words = step.split()
            key_words = []
            word_count = 0
            
            for word in words:
                if len(word) > 3 and word.lower() not in ['the', 'and', 'was', 'were', 'that', 'this', 'with']:
                    key_words.append(word)
                    word_count += 1
                    if word_count >= 4:  # Limit to 4 key words
                        break
            
            if key_words:
                formatted_steps.append(' '.join(key_words))
        
        return formatted_steps if formatted_steps else []
        
    except Exception as e:
        logging.error(f"Error extracting steps: {str(e)}")
        return []

def extract_numerical_data(content: str) -> Dict:
    """Extract numerical data from content."""
    try:
        # Look for numbers in the text
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        categories = ["Item " + str(i+1) for i in range(min(len(numbers), 5))]
        values = [float(num) for num in numbers[:5]]  # Take first 5 numbers
        
        return {
            "categories": categories,
            "values": values,
            "x_axis": "Items",
            "y_axis": "Values"
        }
    except Exception as e:
        logging.error(f"Error extracting numerical data: {str(e)}")
        return {
            "categories": ["Category 1", "Category 2"],
            "values": [1, 1],
            "x_axis": "Categories",
            "y_axis": "Values"
        }

def extract_categorical_data(content: str) -> Dict:
    """Extract categorical data from content."""
    try:
        # Split content into words and count frequencies
        words = content.split()
        from collections import Counter
        word_counts = Counter(words)
        
        # Get top 5 most common words with length > 3
        categories = []
        values = []
        for word, count in word_counts.most_common(10):
            if len(word) > 3 and word.isalpha():
                categories.append(word)
                values.append(count)
            if len(categories) >= 5:
                break
        
        return {
            "categories": categories,
            "values": values
        }
    except Exception as e:
        logging.error(f"Error extracting categorical data: {str(e)}")
        return {
            "categories": ["Category 1", "Category 2"],
            "values": [1, 1]
        }

def extract_timeline_data(content: str) -> Dict:
    """Extract timeline data from content."""
    try:
        # Look for dates in the text
        import re
        dates = re.findall(r'\b\d{4}\b', content)  # Find years
        events = []
        
        for date in dates[:5]:  # Take first 5 dates
            # Get the sentence containing the date
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                if date in sentence:
                    events.append({
                        "date": date,
                        "description": sentence.strip()[:100]  # Limit description length
                    })
                    break
        
        return {
            "timeline_events": events
        }
    except Exception as e:
        logging.error(f"Error extracting timeline data: {str(e)}")
        return {
            "timeline_events": [
                {"date": "2024", "description": "Event 1"},
                {"date": "2025", "description": "Event 2"}
            ]
        }

def extract_concepts_and_relationships(content: str) -> Dict:
    """Extract concepts and their relationships from content."""
    try:
        # Split content into sentences
        sentences = content.split('.')
        
        # Extract key terms (words longer than 3 letters)
        words = [word.strip() for word in content.split() if len(word) > 3]
        from collections import Counter
        word_counts = Counter(words)
        
        # Get top 5 most frequent terms as nodes
        nodes = [word for word, count in word_counts.most_common(5) if word.isalpha()]
        
        # Create simple relationships between consecutive nodes
        relationships = []
        for i in range(len(nodes)-1):
            relationships.append({
                "from": nodes[i],
                "to": nodes[i+1],
                "label": "related to"
            })
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    except Exception as e:
        logging.error(f"Error extracting concepts and relationships: {str(e)}")
        return {
            "nodes": ["Concept 1", "Concept 2"],
            "relationships": [
                {"from": "Concept 1", "to": "Concept 2", "label": "related to"}
            ]
        }

def generate_flowchart(steps: List[str], title: str, style: Dict) -> str:
    """Generate a flowchart visualization with concise steps."""
    try:
        # Validate input steps
        if not steps or not all(isinstance(s, str) and len(s.split()) >= 2 for s in steps):
            logging.warning("Invalid or empty steps provided")
            return ""
            
        # Create a new directed graph
        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB', splines='ortho')
        
        # Set global node and edge styles
        dot.attr('node', 
                shape='box',
                style='rounded,filled',
                fillcolor='#f8f9fa',
                fontname='Arial',
                margin='0.2',
                width='2',
                height='0.5')
        dot.attr('edge',
                fontname='Arial',
                fontsize='10',
                color='#2c3e50')
        
        # Process steps to make them concise but meaningful
        processed_steps = []
        for step in steps:
            # Skip if step is too short or meaningless
            if len(step.split()) < 2 or len(step) < 10:
                continue
                
            # Clean the step text
            step = step.strip()
            # Remove common meaningless prefixes
            step = re.sub(r'^(then|next|after that|finally|lastly)\s*', '', step, flags=re.IGNORECASE)
            # Ensure first letter is capitalized
            step = step[0].upper() + step[1:] if step else step
            
            # Take first 5-7 meaningful words
            words = step.split()
            meaningful_words = []
            for word in words:
                if (len(word) > 2 and 
                    word.lower() not in ['the', 'and', 'was', 'were', 'that', 'this', 'with', 'from', 'using', 'them', 'form']):
                    meaningful_words.append(word)
                if len(meaningful_words) >= 7:
                    break
            
            if len(meaningful_words) >= 2:  # Ensure we have at least 2 meaningful words
                processed_step = ' '.join(meaningful_words)
                # Truncate if still too long
                if len(processed_step) > 40:
                    processed_step = processed_step[:37] + '...'
                processed_steps.append(processed_step)
        
        # Only proceed if we have valid steps
        if not processed_steps:
            logging.warning("No valid steps after processing")
            return ""
        
        # Add nodes and edges
        for i, step in enumerate(processed_steps):
            # Create node
            node_id = f'step_{i}'
            dot.node(node_id, step)
            
            # Connect to previous step
            if i > 0:
                dot.edge(f'step_{i-1}', node_id)
        
        # Save and return
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}")
        dot.render(filename=output_path, cleanup=True, format='png')
        return output_path + '.png'
        
    except Exception as e:
        logging.error(f"Error generating flowchart: {str(e)}")
        return ""

def generate_error_visualization(error_message: str) -> str:
    """Generate an error visualization using matplotlib."""
    try:
        logging.info("Generating error visualization")
        plt.clf()
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add error message
        ax.text(0.5, 0.5, f"Error generating visualization:\n{error_message}",
                ha='center', va='center',
                wrap=True,
                fontsize=12,
                color='red')
        
        # Remove axes
        ax.set_axis_off()
        
        # Save error visualization
        error_path = os.path.join("outputs", "visualization_error.png")
        plt.savefig(error_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        if os.path.exists(error_path):
            logging.info("Error visualization created successfully")
        return error_path
        
        logging.error("Failed to create error visualization")
        return ""
            
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

def generate_concept_map(data: Dict, title: str) -> str:
    """Generate a concept map visualization."""
    try:
        # Create a new directed graph
        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB', splines='ortho')
        
        # Set global node and edge styles
        dot.attr('node',
                shape='box',
                style='rounded,filled',
                fillcolor='#f8f9fa',
                fontname='Arial',
                margin='0.2')
        dot.attr('edge',
                fontname='Arial',
                fontsize='10',
                color='#2c3e50')
        
        # Extract elements
        nodes = data["data"]["nodes"]
        relationships = data["data"]["relationships"]
    
    # Add nodes
        for i, node in enumerate(nodes):
            node_id = f'node_{i}'
            # Truncate long node labels
            node_text = node[:30] + '...' if len(node) > 30 else node
            dot.node(node_id, node_text)
        
        # Add relationships
        for rel in relationships:
            try:
                from_idx = nodes.index(rel["from"])
                to_idx = nodes.index(rel["to"])
                # Truncate long relationship labels
                label = rel["label"][:20] + '...' if len(rel["label"]) > 20 else rel["label"]
                dot.edge(f'node_{from_idx}', f'node_{to_idx}', label)
            except ValueError:
                continue  # Skip if nodes not found
        
        # Save and return
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}")
        dot.render(filename=output_path, cleanup=True, format='png')
        return output_path + '.png'
        
    except Exception as e:
        logging.error(f"Error generating concept map: {str(e)}")
        return generate_error_visualization(str(e))

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

def generate_bar_chart(categories: List[str], values: List[float], title: str, x_label: str, y_label: str) -> str:
    """Generate a bar chart visualization."""
    try:
        plt.figure(figsize=(12, 6))
        
        if not categories or not values:
            return generate_error_visualization("No data for bar chart")
        
        # Create bars
        bars = plt.bar(range(len(categories)), values, color='#3498db')
        
        # Customize the plot
        plt.title(title, pad=20, fontsize=14)
        plt.xlabel(x_label or "Categories")
        plt.ylabel(y_label or "Values")
        
        # Set x-axis labels
        plt.xticks(range(len(categories)), 
                  [c[:20] + '...' if len(c) > 20 else c for c in categories],
                  rotation=45,
                  ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:g}',
                    ha='center', va='bottom')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and return
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error generating bar chart: {str(e)}")
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

def generate_timeline(events: List[Dict], title: str) -> str:
    """Generate a timeline visualization."""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create new figure
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        
        if not events:
            plt.close(fig)
            return generate_error_visualization("No events for timeline")
        
        # Extract dates and descriptions
        dates = []
        descriptions = []
        for event in events:
            dates.append(event["date"])
            descriptions.append(event["description"])
        
        # Create timeline
        levels = np.zeros(len(dates))
        for i in range(len(dates)):
            if i > 0:
                levels[i] = (levels[i-1] + 1) % 2
        
        # Plot events
        markerline, stemline, baseline = ax.stem(dates, levels)
        plt.setp(markerline, marker='o', markersize=10, markerfacecolor='#3498db')
        plt.setp(stemline, color='#95a5a6', linestyle='--')
        
        # Add descriptions
        for i, (date, desc, level) in enumerate(zip(dates, descriptions, levels)):
            ax.annotate(desc, xy=(date, level), xytext=(10, 10 if level else -10),
                        textcoords='offset points',
                        ha='left',
                       va='bottom' if level else 'top',
                       wrap=True)
        
        # Customize the plot
        ax.set_title(title, pad=20, fontsize=14)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save and return
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path
    except Exception as e:
        logging.error(f"Error generating timeline: {str(e)}")
        logging.error(traceback.format_exc())
        return generate_error_visualization(str(e))

def analyze_results_for_visualization(text: str) -> Dict:
    """Analyze results section to extract numerical data and relationships."""
    prompt = (
        "Extract key numerical findings and relationships from this Results section for visualization.\n\n"
        f"Content: {text}\n\n"
        "Provide a structured JSON response with:\n"
        "{\n"
        '    "chart_type": "bar|line|scatter|pie",\n'
        '    "title": "Main finding or comparison",\n'
        '    "data": {\n'
        '        "categories": ["category1", "category2"],\n'
        '        "values": [value1, value2],\n'
        '        "labels": ["label1", "label2"],\n'
        '        "comparisons": [{"group1": "name1", "value1": num1, "group2": "name2", "value2": num2}]\n'
        '    },\n'
        '    "axis_labels": {"x": "x-axis label", "y": "y-axis label"},\n'
        '    "highlight_points": ["key findings to emphasize"]\n'
        "}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data visualization expert that extracts numerical insights from research results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    except Exception as e:
        logging.error(f"Error analyzing results: {str(e)}")
        return {
            "chart_type": "bar",
            "title": "Key Findings",
            "data": {"categories": [], "values": [], "labels": [], "comparisons": []},
            "axis_labels": {"x": "", "y": ""}
        }

def generate_results_visualization(data: Dict, title: str) -> str:
    """Generate statistical visualization for results section."""
    try:
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn')
        
        chart_type = data.get("chart_type", "bar")
        plot_data = data.get("data", {})
        categories = plot_data.get("categories", [])
        values = plot_data.get("values", [])
        comparisons = plot_data.get("comparisons", [])
        
        if chart_type == "bar":
            # Create grouped bar chart for comparisons
            if comparisons:
                groups = [comp["group1"] for comp in comparisons]
                values1 = [comp["value1"] for comp in comparisons]
                values2 = [comp["value2"] for comp in comparisons]
                
                x = np.arange(len(groups))
                width = 0.35
                
                plt.bar(x - width/2, values1, width, label=comparisons[0]["group1"], color='#2ecc71')
                plt.bar(x + width/2, values2, width, label=comparisons[0]["group2"], color='#3498db')
                plt.xticks(x, groups, rotation=45)
                plt.legend()
            
            # Simple bar chart for single values
            elif categories and values:
                plt.bar(categories, values, color='#3498db')
                plt.xticks(rotation=45)
        
        elif chart_type == "line":
            plt.plot(categories, values, marker='o', linewidth=2, color='#2ecc71')
            plt.xticks(rotation=45)
        
        elif chart_type == "scatter":
            if len(values) >= 2:
                x_vals = values[::2]
                y_vals = values[1::2]
                plt.scatter(x_vals, y_vals, alpha=0.6, color='#3498db')
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                plt.plot(x_vals, p(x_vals), "r--", alpha=0.8)
        
        elif chart_type == "pie":
            plt.pie(values, labels=categories, autopct='%1.1f%%', 
                   colors=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6'])
            plt.axis('equal')
        
        # Add labels and title
        plt.xlabel(data.get("axis_labels", {}).get("x", ""))
        plt.ylabel(data.get("axis_labels", {}).get("y", ""))
        plt.title(data.get("title", title), pad=20, fontsize=14)
        
        # Add grid for non-pie charts
        if chart_type != "pie":
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    except Exception as e:
        logging.error(f"Error generating results visualization: {str(e)}")
        return generate_error_visualization(str(e))

def validate_and_format_data(viz_data: Dict) -> Dict:
    """Validate and format visualization data."""
    try:
        # Default structure
        default_data = {
            "viz_type": "BAR_CHART",
            "title": "Visualization",
            "explanation": "",
            "data": {
                "categories": [],
                "values": [],
                "x_axis": "",
                "y_axis": ""
            },
            "style": {
                "colors": ["#3498db"],
                "layout": "vertical"
            }
        }

        if not isinstance(viz_data, dict):
            logging.error("Invalid visualization data format")
            return default_data

        # Extract and validate visualization type
        viz_type = viz_data.get("viz_type", "").upper()
        if viz_type not in ["CONCEPT_MAP", "FLOWCHART", "BAR_CHART", "LINE_CHART", 
                           "SCATTER_PLOT", "PIE_CHART", "HIERARCHY_TREE", "TIMELINE"]:
            logging.warning(f"Invalid visualization type: {viz_type}, defaulting to BAR_CHART")
            viz_type = "BAR_CHART"

        # Extract data based on visualization type
        data = viz_data.get("data", {})
        if not isinstance(data, dict):
            data = {}

        if viz_type in ["BAR_CHART", "LINE_CHART", "PIE_CHART"]:
            # Ensure categories and values are lists
            categories = data.get("categories", [])
            values = data.get("values", [])
            
            if not isinstance(categories, list) or not isinstance(values, list):
                categories = []
                values = []
            
            # Convert values to float where possible
            values = [float(v) if isinstance(v, (int, float, str)) and str(v).replace('.','').isdigit() else 0 
                     for v in values]
            
            formatted_data = {
                "categories": categories,
                "values": values,
                "x_axis": str(data.get("x_axis", "")),
                "y_axis": str(data.get("y_axis", ""))
            }
            
        elif viz_type == "CONCEPT_MAP":
            formatted_data = {
                "nodes": [str(n) for n in data.get("nodes", []) if n],
                "relationships": [
                    {
                        "from": str(r.get("from", "")),
                        "to": str(r.get("to", "")),
                        "label": str(r.get("label", ""))
                    }
                    for r in data.get("relationships", [])
                    if isinstance(r, dict) and r.get("from") and r.get("to")
                ]
            }
            
        elif viz_type == "HIERARCHY_TREE":
            hierarchy = data.get("hierarchy", {})
            if not isinstance(hierarchy, dict):
                hierarchy = {}
            
            formatted_data = {
                "hierarchy": {
                    "root": str(hierarchy.get("root", "Main Topic")),
                    "children": [str(c) for c in hierarchy.get("children", []) if c]
                }
            }
            
        elif viz_type == "TIMELINE":
            events = data.get("timeline_events", [])
            formatted_data = {
                "timeline_events": [
                    {
                        "date": str(e.get("date", "")),
                        "description": str(e.get("description", ""))
                    }
                    for e in events
                    if isinstance(e, dict) and e.get("date") and e.get("description")
                ]
            }
        else:
            formatted_data = data

        # Format style
        style = viz_data.get("style", {})
        if not isinstance(style, dict):
            style = {}

        formatted_style = {
            "colors": style.get("colors", ["#3498db"]),
            "layout": style.get("layout", "vertical")
        }

        return {
            "viz_type": viz_type,
            "title": str(viz_data.get("title", "Visualization")),
            "explanation": str(viz_data.get("explanation", "")),
            "data": formatted_data,
            "style": formatted_style
        }

    except Exception as e:
        logging.error(f"Error in data validation: {str(e)}")
        return default_data

def generate_visualization(viz_data: Dict) -> str:
    """Generate visualization based on the analysis results."""
    try:
        logging.info(f"Starting visualization generation for type: {viz_data['viz_type']}")
        
        # Skip if no visualization needed
        if viz_data["viz_type"] == "NONE":
            logging.info("No visualization required")
            return ""
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Reset matplotlib state
        plt.clf()
        plt.close('all')
        
        if viz_data["viz_type"] == "FLOWCHART":
            if not viz_data["data"].get("steps"):
                return ""
            return generate_flowchart(
                viz_data["data"]["steps"],
                viz_data["title"],
                viz_data["style"]
            )
        elif viz_data["viz_type"] == "BAR_CHART":
            if not viz_data["data"].get("values"):
                return ""
            return generate_bar_chart(
                viz_data["data"]["categories"],
                viz_data["data"]["values"],
                viz_data["title"],
                viz_data["data"].get("x_axis", ""),
                viz_data["data"].get("y_axis", "")
            )
        else:
            logging.info(f"Unsupported visualization type: {viz_data['viz_type']}")
            return ""
            
    except Exception as e:
        logging.error(f"Error in generate_visualization: {str(e)}")
        logging.error(traceback.format_exc())
        return ""

def generate_hierarchy_tree(hierarchy: Dict, title: str) -> str:
    """Generate a hierarchical tree visualization."""
    try:
        dot = graphviz.Digraph(comment=title)
        dot.attr(rankdir='TB')
        
        def add_nodes(parent: str, children: List, prefix: str = ""):
            if not isinstance(children, list):
                return
            
            for i, child in enumerate(children):
                node_id = f"{prefix}{i}"
                dot.node(node_id, child)
                dot.edge(parent, node_id)
        
        # Add root
        root = hierarchy.get("root", "Main Topic")
        dot.node("root", root, shape="doubleoctagon")
        
        # Add children
        add_nodes("root", hierarchy.get("children", []), "child_")
        
        # Save and return
        output_path = os.path.join("outputs", f"visualization_{title.lower().replace(' ', '_')}")
        dot.render(filename=output_path, cleanup=True, format='png')
        return output_path + '.png'
    except Exception as e:
        return generate_error_visualization(str(e))

def visualize_section(title: str, section_text: str) -> str:
    """Visualize a section based on intelligent content analysis."""
    try:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Analyze content and determine best visualization
        viz_data = analyze_content_for_visualization(title, section_text)
        
        # Generate and return visualization
        return generate_visualization(viz_data)
            
    except Exception as e:
        error_msg = f"Error in visualize_section: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return generate_error_visualization(error_msg)
