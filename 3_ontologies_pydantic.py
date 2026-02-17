#!/usr/bin/env python3
"""
Demo: Create knowledge graphs dynamically using Pydantic and LLMs.

This script demonstrates how to:
1. Extract text from a PDF document
2. Use an LLM to identify concrete entities and their relationships
3. Build a knowledge graph with actual instances (not just a schema)
4. Visualize the graph in multiple formats (JSON, DOT/Graphviz, text)

Key difference from 2_ontologies_owl.py:
- Script 2: Extracts ontology **SCHEMA** (classes and properties) → OWL file
- Script 3: Extracts **INSTANCES** (actual entities and relationships) → Knowledge graph

The output is a concrete knowledge graph showing real entities from the document
and how they connect to each other, making it perfect for presentation slides.

Requires GOOGLE_API_KEY in the environment.
"""
import argparse
import json
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils import get_llm, extract_text_from_pdf


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("knowledge_graphs")
DEFAULT_PDF = "cssanalyse295-en.pdf"


# ============================================================================
# Pydantic Models for Knowledge Graph Instances
# ============================================================================
# These models represent actual entities and relationships in the knowledge graph.
# Unlike ontology schemas, these are concrete instances extracted from the document.


class Entity(BaseModel):
    """
    A concrete entity (instance) in the knowledge graph.

    Examples:
    - Person: "John Doe", "Marie Curie"
    - Organization: "NASA", "Google"
    - Location: "Paris", "Silicon Valley"
    - Concept: "Machine Learning", "Climate Change"
    - Event: "World War II", "2024 Olympics"
    """
    id: str = Field(
        description="Unique identifier for this entity (use snake_case, e.g., 'nasa_organization')"
    )
    type: str = Field(
        description="The type/category of this entity (e.g., 'Person', 'Organization', 'Technology')"
    )
    label: str = Field(
        description="Human-readable name/label for this entity"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties as key-value pairs (e.g., {'founded': '1958', 'location': 'USA'})"
    )


class Relationship(BaseModel):
    """
    A relationship between two entities in the knowledge graph.

    Examples:
    - "John Doe" --[works_at]--> "NASA"
    - "Machine Learning" --[is_part_of]--> "Artificial Intelligence"
    - "Paris" --[capital_of]--> "France"
    """
    source_id: str = Field(
        description="The ID of the source entity (subject of the relationship)"
    )
    relation_type: str = Field(
        description="The type of relationship (use snake_case, e.g., 'works_at', 'founded_by', 'located_in')"
    )
    target_id: str = Field(
        description="The ID of the target entity (object of the relationship)"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Optional metadata about this relationship (e.g., {'since': '2020', 'confidence': 'high'})"
    )


class KnowledgeGraph(BaseModel):
    """
    A complete knowledge graph with entities and relationships.

    This represents the structured knowledge extracted from a document,
    showing both what entities exist and how they're connected.
    """
    name: str = Field(
        description="A descriptive name for this knowledge graph"
    )
    description: str = Field(
        description="A brief summary of what this graph represents"
    )
    entities: List[Entity] = Field(
        description="All entities (nodes) in the graph"
    )
    relationships: List[Relationship] = Field(
        description="All relationships (edges) between entities"
    )


# ============================================================================
# LLM Prompt for Knowledge Graph Extraction
# ============================================================================

KNOWLEDGE_GRAPH_EXTRACTION_PROMPT = """You are an expert in knowledge graph construction and information extraction.

Your task is to analyze a document and extract a knowledge graph with concrete entities and their relationships.

Guidelines for extraction:

1. **Entities**: Identify specific, concrete entities mentioned in the document.
   - Use clear, unique IDs (snake_case): e.g., 'renesas_company', 'n3_building', 'automotive_industry'
   - Assign intuitive types: 'Organization', 'Location', 'Technology', 'Event', 'Industry', 'Product', etc.
   - Use the actual name from the document as the label
   - Add key properties mentioned in the text (dates, descriptions, attributes)

2. **Relationships**: Identify how entities connect to each other.
   - Use descriptive relation types (snake_case): e.g., 'produces', 'located_in', 'affected_by', 'supplies_to'
   - Only create relationships explicitly stated or strongly implied in the text
   - Ensure source_id and target_id match entity IDs exactly

3. **Keep it demo-friendly**:
   - Aim for 8-15 entities (enough to be interesting, not overwhelming)
   - Aim for 10-20 relationships (shows meaningful connections)
   - Focus on the most important concepts and their key relationships
   - Prioritize clarity: this is for a presentation slide deck

4. **Be specific to the document**:
   - Extract real entities mentioned in the text, not generic placeholders
   - Use actual names, organizations, technologies, etc. from the document
   - Capture domain-specific terminology

Return a structured knowledge graph (entities and relationships)."""


def extract_knowledge_graph_with_llm(
    text: str,
) -> KnowledgeGraph:
    """
    Use an LLM to extract a knowledge graph from text.

    This function prompts an LLM to identify concrete entities and relationships
    in the text, then parses the response into a structured KnowledgeGraph object.

    Args:
        text: The input text to analyze

    Returns:
        KnowledgeGraph: The extracted knowledge graph with entities and relationships
    """
    # Initialize the LLM with low temperature and bind structured output
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(KnowledgeGraph)

    messages = [
        SystemMessage(content=KNOWLEDGE_GRAPH_EXTRACTION_PROMPT),
        HumanMessage(content=f"""Analyze the following document and extract a knowledge graph with entities and relationships.

Document text:
---
{text}
---""")
    ]

    # Invoke the LLM; with_structured_output returns a KnowledgeGraph directly
    print("  Calling LLM to extract knowledge graph...")
    try:
        kg = structured_llm.invoke(messages)
        print(f"  ✓ Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
        return kg
    except Exception as e:
        print(f"\nERROR: Failed to extract knowledge graph: {e}")
        raise


# ============================================================================
# Visualization Functions
# ============================================================================

def save_as_json(kg: KnowledgeGraph, output_path: Path) -> None:
    """
    Save the knowledge graph as a pretty-printed JSON file.

    This format is:
    - Human-readable and easy to inspect
    - Can be loaded by any programming language
    - Good for debugging and documentation

    Args:
        kg: The knowledge graph to save
        output_path: Where to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"  ✓ Knowledge graph saved as JSON: {output_path}")


def save_as_dot(kg: KnowledgeGraph, output_path: Path) -> None:
    """
    Save the knowledge graph in DOT format (Graphviz).

    DOT is a graph description language that can be:
    - Rendered as images (PNG, SVG, PDF) using Graphviz tools
    - Visualized interactively in graph viewers
    - Embedded in documentation and presentations

    To render: `dot -Tpng graph.dot -o graph.png`
    Or use online viewers: http://www.webgraphviz.com/

    Args:
        kg: The knowledge graph to save
        output_path: Where to save the .dot file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the DOT file content
    lines = [
        "digraph KnowledgeGraph {",
        "  // Graph styling for better visualization",
        "  graph [rankdir=LR, bgcolor=white, fontname=\"Arial\"];",
        "  node [shape=box, style=filled, fillcolor=lightblue, fontname=\"Arial\"];",
        "  edge [fontname=\"Arial\", fontsize=10];",
        "",
        "  // Entities (nodes)",
    ]

    # Add all entities as nodes
    for entity in kg.entities:
        # Create a label with entity type and name
        label = f"{entity.label}\\n({entity.type})"
        # Add key properties to the label if present
        if entity.properties:
            # Show first 2 properties to avoid cluttering
            props = list(entity.properties.items())[:2]
            prop_lines = [f"{k}: {v}" for k, v in props]
            label += "\\n" + "\\n".join(prop_lines)

        # Escape quotes in the label
        label = label.replace('"', '\\"')
        lines.append(f'  "{entity.id}" [label="{label}"];')

    lines.append("")
    lines.append("  // Relationships (edges)")

    # Add all relationships as edges
    for rel in kg.relationships:
        # Create edge label from relation type
        label = rel.relation_type.replace("_", " ")
        lines.append(f'  "{rel.source_id}" -> "{rel.target_id}" [label="{label}"];')

    lines.append("}")

    # Write the DOT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Knowledge graph saved as DOT: {output_path}")
    print(f"    To visualize: dot -Tpng {output_path} -o {output_path.with_suffix('.png')}")
    print(f"    Or use: http://www.webgraphviz.com/")


def print_text_visualization(kg: KnowledgeGraph) -> None:
    """
    Print a human-readable text representation of the knowledge graph.

    This is perfect for:
    - Quick inspection during development
    - Including in presentation notes
    - Demonstrating the structure without visual tools

    Args:
        kg: The knowledge graph to visualize
    """
    width = 72
    line = "═" * width
    thin = "─" * width

    print(f"\n{line}")
    print(f"  KNOWLEDGE GRAPH: {kg.name}")
    print(thin)
    print(f"  {kg.description}")
    print(line)

    # Print entities
    print(f"\n  ENTITIES ({len(kg.entities)} nodes)")
    print(thin)
    for entity in kg.entities:
        print(f"\n  [{entity.id}]")
        print(f"    Type:  {entity.type}")
        print(f"    Label: {entity.label}")
        if entity.properties:
            print(f"    Properties:")
            for key, value in entity.properties.items():
                print(f"      • {key}: {value}")

    # Print relationships
    print(f"\n{thin}")
    print(f"  RELATIONSHIPS ({len(kg.relationships)} edges)")
    print(thin)

    # Create a lookup for entity labels
    entity_labels = {e.id: e.label for e in kg.entities}

    for rel in kg.relationships:
        source_label = entity_labels.get(rel.source_id, rel.source_id)
        target_label = entity_labels.get(rel.target_id, rel.target_id)
        relation = rel.relation_type.replace("_", " ")

        print(f"\n  {source_label}")
        print(f"    --[{relation}]-->")
        print(f"  {target_label}")

        if rel.properties:
            print(f"    (properties: {rel.properties})")

    print(f"\n{line}\n")


# ============================================================================
# Main Script
# ============================================================================

def main():
    """
    Main entry point for the knowledge graph extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph from PDF using LLM and Pydantic"
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default=DEFAULT_PDF,
        help=f"PDF file to analyze (default: {DEFAULT_PDF})"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output filename (without extension, default: based on graph name)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        help="First page to extract (0-indexed)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        help="Last page to extract (0-indexed, inclusive)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "dot", "both"],
        default="both",
        help="Output format (default: both)"
    )
    args = parser.parse_args()

    # ========================================================================
    # Step 1: Load the PDF
    # ========================================================================
    pdf_path = DATA_DIR / args.pdf_file
    if not pdf_path.exists():
        # Try absolute path if not found in data directory
        pdf_path = Path(args.pdf_file)
        if not pdf_path.exists():
            raise SystemExit(f"ERROR: PDF file not found: {args.pdf_file}")

    print(f"\n{'='*72}")
    print(f"  KNOWLEDGE GRAPH EXTRACTION FROM PDF")
    print(f"{'='*72}")
    print(f"\n  Input PDF: {pdf_path}")

    # Extract text from the PDF
    print(f"  Extracting text from PDF...")
    text = extract_text_from_pdf(
        pdf_path,
        start_page=args.start_page,
        end_page=args.end_page
    )
    print(f"  ✓ Extracted {len(text)} characters")

    # ========================================================================
    # Step 2: Extract knowledge graph using LLM
    # ========================================================================
    print(f"\n{'─'*72}")
    print("  KNOWLEDGE GRAPH EXTRACTION")
    print(f"{'─'*72}\n")

    kg = extract_knowledge_graph_with_llm(text)

    # ========================================================================
    # Step 3: Display text visualization
    # ========================================================================
    print_text_visualization(kg)

    # ========================================================================
    # Step 4: Save output files
    # ========================================================================
    print(f"{'─'*72}")
    print("  SAVING OUTPUT FILES")
    print(f"{'─'*72}\n")

    # Determine base filename
    if args.output:
        base_name = args.output
    else:
        # Use graph name from LLM, cleaned up
        base_name = kg.name.lower().replace(" ", "_")

    # Save in requested format(s)
    if args.format in ["json", "both"]:
        json_path = OUTPUT_DIR / f"{base_name}.json"
        save_as_json(kg, json_path)

    if args.format in ["dot", "both"]:
        dot_path = OUTPUT_DIR / f"{base_name}.dot"
        save_as_dot(kg, dot_path)

    # ========================================================================
    # Final instructions
    # ========================================================================
    print(f"\n{'='*72}")
    print(f"  DONE!")
    print(f"{'='*72}\n")
    print("  Next steps:")
    print(f"  1. View the JSON file for a structured representation")

    if args.format in ["dot", "both"]:
        dot_path = OUTPUT_DIR / f"{base_name}.dot"
        png_path = dot_path.with_suffix(".png")
        print(f"  2. Visualize the graph:")
        print(f"     dot -Tpng {dot_path} -o {png_path}")
        print(f"     (requires Graphviz: apt install graphviz)")
        print(f"  3. Or paste the .dot file content into: http://www.webgraphviz.com/")

    print(f"\n  This knowledge graph is ready for your presentation slides!\n")


if __name__ == "__main__":
    main()
