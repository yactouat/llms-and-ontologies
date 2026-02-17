#!/usr/bin/env python3
"""
Demo: Create OWL ontologies dynamically using an LLM.

This script demonstrates how to:
1. Extract text from a PDF document
2. Use an LLM to identify concepts, relationships, and properties
3. Generate an OWL ontology file readable in Protégé

The output is a knowledge graph in OWL/RDF format that can be:
- Visualized in Protégé (https://protege.stanford.edu/)
- Queried with SPARQL
- Reasoned over with OWL reasoners
- Integrated with other semantic web technologies

Requires GOOGLE_API_KEY in the environment.
"""
import argparse
import json
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from owlready2 import (
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    Thing,
    get_ontology,
)
from pydantic import BaseModel, Field

from utils import get_llm, extract_text_from_pdf


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("ontologies")
DEFAULT_PDF = "cssanalyse295-en.pdf"


# ============================================================================
# Pydantic Models for Structured LLM Output
# ============================================================================
# These models define the structure we want the LLM to extract from the text.
# The LangChain PydanticOutputParser will format the LLM prompt to match
# these schemas and parse the response into typed Python objects.


class OntologyClass(BaseModel):
    """
    Represents a class (concept) in the ontology.

    In OWL, classes represent sets of individuals (instances).
    For example: Person, Document, Organization, Event, etc.
    """
    name: str = Field(
        description="The name of the class (use PascalCase, e.g., 'SecurityThreat')"
    )
    comment: str = Field(
        description="A human-readable description of what this class represents"
    )


class OntologyDataProperty(BaseModel):
    """
    Represents a data property in the ontology.

    Data properties link individuals to literal values (strings, numbers, dates, etc.).
    For example: hasName, hasAge, hasDate, hasDescription
    """
    name: str = Field(
        description="The name of the data property (use camelCase, e.g., 'hasTitle')"
    )
    comment: str = Field(
        description="A human-readable description of what this property represents"
    )
    domain: List[str] = Field(
        description="List of class names that can have this property"
    )
    range: str = Field(
        description="The data type (e.g., 'string', 'int', 'float', 'bool', 'date')"
    )


class OntologyObjectProperty(BaseModel):
    """
    Represents an object property in the ontology.

    Object properties link individuals to other individuals.
    For example: hasAuthor, isPartOf, relatesTo, dependsOn
    """
    name: str = Field(
        description="The name of the object property (use camelCase, e.g., 'hasAuthor')"
    )
    comment: str = Field(
        description="A human-readable description of what this relationship represents"
    )
    domain: List[str] = Field(
        description="List of class names that can have this property (subject)"
    )
    range: List[str] = Field(
        description="List of class names this property can point to (object)"
    )
    is_functional: bool = Field(
        default=False,
        description="True if this property can only have one value (e.g., hasDirectSupervisor)"
    )


class ExtractedOntology(BaseModel):
    """
    Complete ontology structure extracted from a document.

    This represents the knowledge graph schema that will be saved as OWL.
    """
    name: str = Field(
        description="A short name for the ontology (no spaces, use underscores)"
    )
    description: str = Field(
        description="A brief description of what domain this ontology covers"
    )
    classes: List[OntologyClass] = Field(
        description="The main concepts/entities in the domain"
    )
    data_properties: List[OntologyDataProperty] = Field(
        description="Properties that link entities to literal values"
    )
    object_properties: List[OntologyObjectProperty] = Field(
        description="Properties that link entities to other entities"
    )


# ============================================================================
# LLM Prompt for Ontology Extraction
# ============================================================================

ONTOLOGY_EXTRACTION_SYSTEM_PROMPT = """You are an expert in knowledge representation and ontology engineering.

Your task is to analyze a document and extract a well-structured ontology (knowledge graph schema) from it.

Guidelines for extraction:

1. **Classes**: Identify the main concepts, entities, or types mentioned in the document.
   - Use clear, descriptive PascalCase names (e.g., SecurityThreat, AttackVector)
   - Focus on the most important 5-10 concepts to keep the ontology manageable
   - Each class should represent a distinct type of thing

2. **Data Properties**: Identify attributes that describe entities with literal values.
   - Use camelCase naming (e.g., hasTitle, hasDate, hasDescription)
   - Specify appropriate data types (string, int, float, bool, date)
   - Only include properties that are explicitly or implicitly mentioned

3. **Object Properties**: Identify relationships between entities.
   - Use camelCase naming (e.g., targets, exploits, mitigates)
   - Mark as functional if the property can only have one value per instance
   - Focus on meaningful relationships that capture domain knowledge

4. **Keep it simple**: This is a demo ontology for educational purposes.
   - Aim for 5-10 classes, 5-15 data properties, and 5-10 object properties
   - Prioritize clarity over completeness
   - Use intuitive, self-documenting names

5. **Be domain-specific**: Extract concepts relevant to the document's subject matter.
   - Don't create generic ontologies that could apply to any document
   - Capture the specific terminology and relationships in this domain

Your output must be valid JSON matching the specified schema."""


def extract_ontology_with_llm(
    text: str,
    max_chars: int = 8000
) -> ExtractedOntology:
    """
    Use an LLM to extract ontology structure from text.

    This function:
    1. Truncates the text if needed (LLMs have context limits)
    2. Creates a structured prompt with the extraction guidelines
    3. Calls the LLM to analyze the text
    4. Parses the LLM response into a structured Pydantic model

    Args:
        text: The input text to analyze
        max_chars: Maximum characters to send to LLM (prevents context overflow)

    Returns:
        ExtractedOntology: The structured ontology extracted from the text
    """
    # Truncate text if too long (leave room for prompt and response)
    if len(text) > max_chars:
        print(f"  Text truncated to {max_chars} characters for LLM processing")
        text = text[:max_chars] + "\n\n[... truncated ...]"

    # Initialize the LLM (temperature=0 for deterministic output)
    llm = get_llm(temperature=0.0)

    # Create a parser that will format the prompt and parse the response
    parser = PydanticOutputParser(pydantic_object=ExtractedOntology)

    # Build the prompt with format instructions
    messages = [
        SystemMessage(content=ONTOLOGY_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze the following document and extract an ontology.

Document text:
---
{text}
---

{parser.get_format_instructions()}

Extract the ontology in the required JSON format:""")
    ]

    # Call the LLM
    print("  Calling LLM to extract ontology structure...")
    response = llm.invoke(messages)

    # Extract content from response
    content = response.content if hasattr(response, "content") else str(response)

    # Handle potential list response format
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)

    # Parse the LLM response into our Pydantic model
    try:
        ontology = parser.parse(content)
        print(f"  ✓ Extracted {len(ontology.classes)} classes, "
              f"{len(ontology.data_properties)} data properties, "
              f"{len(ontology.object_properties)} object properties")
        return ontology
    except Exception as e:
        print(f"\nERROR: Failed to parse LLM response: {e}")
        print("\nLLM Response:")
        print(content)
        raise


def create_owl_ontology(
    extracted: ExtractedOntology,
    output_path: Path
) -> None:
    """
    Create an OWL ontology file from the extracted structure.

    This function uses owlready2 to:
    1. Create an ontology with a unique IRI (Internationalized Resource Identifier)
    2. Define all classes as subclasses of owl:Thing
    3. Define all properties with their domains and ranges
    4. Save the ontology in RDF/XML format (readable by Protégé)

    Args:
        extracted: The ontology structure extracted by the LLM
        output_path: Where to save the .owl file
    """
    # Create a unique IRI (identifier) for this ontology
    # In real ontologies, this would be a resolvable URL you control
    ontology_iri = f"http://example.com/ontologies/{extracted.name}"

    print(f"\n  Creating OWL ontology: {ontology_iri}")

    # Initialize the ontology
    # owlready2 uses a "with" context to define ontology contents
    onto = get_ontology(ontology_iri)

    with onto:
        # ====================================================================
        # Step 1: Create all classes
        # ====================================================================
        # In OWL, classes are defined as Python classes that inherit from Thing
        # We store them in a dict to reference them when creating properties
        created_classes = {}

        print(f"  Creating {len(extracted.classes)} classes...")
        for cls in extracted.classes:
            # Dynamically create a Python class for each OWL class
            # The class name becomes the OWL class name
            # The comment becomes rdfs:comment in the ontology
            owl_class = type(
                cls.name,  # Class name
                (Thing,),  # Inherit from owl:Thing
                {
                    "comment": [cls.comment],  # rdfs:comment annotation
                }
            )
            created_classes[cls.name] = owl_class
            print(f"    - {cls.name}")

        # ====================================================================
        # Step 2: Create data properties
        # ====================================================================
        # Data properties link individuals to literal values
        print(f"\n  Creating {len(extracted.data_properties)} data properties...")
        for prop in extracted.data_properties:
            # Get the domain classes (what can have this property)
            domain_classes = [
                created_classes[cls_name]
                for cls_name in prop.domain
                if cls_name in created_classes
            ]

            # Map the range string to Python types
            # owlready2 uses Python types for data property ranges
            range_type = _map_data_range(prop.range)

            # Create the data property
            type(
                prop.name,
                (DataProperty,),
                {
                    "comment": [prop.comment],
                    "domain": domain_classes,
                    "range": [range_type],
                }
            )
            print(f"    - {prop.name} (domain: {prop.domain}, range: {prop.range})")

        # ====================================================================
        # Step 3: Create object properties
        # ====================================================================
        # Object properties link individuals to other individuals
        print(f"\n  Creating {len(extracted.object_properties)} object properties...")
        for prop in extracted.object_properties:
            # Get domain and range classes
            domain_classes = [
                created_classes[cls_name]
                for cls_name in prop.domain
                if cls_name in created_classes
            ]
            range_classes = [
                created_classes[cls_name]
                for cls_name in prop.range
                if cls_name in created_classes
            ]

            # Determine the base classes for the property
            # Functional properties can have at most one value per individual
            if prop.is_functional:
                bases = (ObjectProperty, FunctionalProperty)
                func_marker = " [functional]"
            else:
                bases = (ObjectProperty,)
                func_marker = ""

            # Create the object property
            type(
                prop.name,
                bases,
                {
                    "comment": [prop.comment],
                    "domain": domain_classes,
                    "range": range_classes,
                }
            )
            print(f"    - {prop.name}{func_marker} (domain: {prop.domain}, range: {prop.range})")

    # ====================================================================
    # Step 4: Save the ontology
    # ====================================================================
    # Save in RDF/XML format (the standard format for OWL ontologies)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(output_path), format="rdfxml")

    print(f"\n  ✓ Ontology saved to: {output_path}")
    print(f"\n  You can now open this file in Protégé to visualize the knowledge graph!")


def _map_data_range(range_str: str):
    """
    Map string data type names to Python types for owlready2.

    Args:
        range_str: The data type as a string (e.g., "string", "int", "float")

    Returns:
        Python type object or str as default
    """
    type_map = {
        "string": str,
        "str": str,
        "int": int,
        "integer": int,
        "float": float,
        "double": float,
        "bool": bool,
        "boolean": bool,
        # For dates, owlready2 can use datetime, but str is safer for demos
        "date": str,
        "datetime": str,
    }
    return type_map.get(range_str.lower(), str)


def main():
    """
    Main entry point for the ontology generation script.
    """
    parser = argparse.ArgumentParser(
        description="Generate OWL ontology from PDF using LLM analysis"
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default=DEFAULT_PDF,
        help=f"PDF file to analyze (default: {DEFAULT_PDF})"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .owl file name (default: based on ontology name from LLM)"
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
    print(f"  ONTOLOGY GENERATION FROM PDF")
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
    # Step 2: Extract ontology structure using LLM
    # ========================================================================
    print(f"\n{'─'*72}")
    print("  ONTOLOGY EXTRACTION")
    print(f"{'─'*72}\n")

    extracted = extract_ontology_with_llm(text)

    print(f"\n  Ontology: {extracted.name}")
    print(f"  Description: {extracted.description}")

    # ========================================================================
    # Step 3: Create OWL file
    # ========================================================================
    print(f"\n{'─'*72}")
    print("  OWL GENERATION")
    print(f"{'─'*72}\n")

    # Determine output filename
    if args.output:
        output_path = OUTPUT_DIR / args.output
    else:
        output_path = OUTPUT_DIR / f"{extracted.name.lower()}.owl"

    # Ensure .owl extension
    if not output_path.suffix == ".owl":
        output_path = output_path.with_suffix(".owl")

    create_owl_ontology(extracted, output_path)

    # ========================================================================
    # Step 4: Save a human-readable summary
    # ========================================================================
    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(extracted.model_dump(), f, indent=2)

    print(f"  ✓ Human-readable summary saved to: {summary_path}")

    print(f"\n{'='*72}")
    print(f"  DONE!")
    print(f"{'='*72}\n")
    print("  Next steps:")
    print("  1. Download Protégé from https://protege.stanford.edu/")
    print(f"  2. Open the file: {output_path}")
    print("  3. Explore the Classes, Object Properties, and Data Properties tabs")
    print("  4. Use the OntoGraf tab to visualize the knowledge graph\n")


if __name__ == "__main__":
    main()
