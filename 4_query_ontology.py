#!/usr/bin/env python3
"""
Demo: Query knowledge graphs dynamically to answer complex questions.

This script demonstrates how to:
1. Load multiple PDF documents from data/
2. Extract knowledge graphs from each using LLM
3. Merge knowledge graphs into a unified graph
4. Query the graph to answer complex multi-hop questions
5. Show superiority over vector search for reasoning tasks

Key advantages over vector search (1_vector_search.py):
- Can traverse relationships between entities (multi-hop reasoning)
- Can link information across documents through shared entities
- Can answer causal questions by following relationship chains
- Can handle temporal reasoning by traversing time-based relationships

Requires GOOGLE_API_KEY in the environment.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils import get_llm, extract_text_from_pdf, load_pdfs_from_directory


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
CACHE_DIR = Path("knowledge_graphs_cache")
DEFAULT_QUESTION = "How did the N3 Building fire affect Kentucky Truck Plant inventory?"


# ============================================================================
# Pydantic Models (same as 3_ontologies_pydantic.py)
# ============================================================================
# These models represent entities and relationships in knowledge graphs.
# They are used both for extraction and for querying.


class Entity(BaseModel):
    """
    A concrete entity (instance) in the knowledge graph.

    Examples:
    - Organization: "Renesas", "Ford Motor Company"
    - Location: "N3 Building", "Kentucky Truck Plant"
    - Product: "RH850 MCU", "F-150 truck"
    - Event: "Factory fire", "Production halt"
    """
    id: str = Field(
        description="Unique identifier for this entity (use snake_case)"
    )
    type: str = Field(
        description="The type/category of this entity"
    )
    label: str = Field(
        description="Human-readable name/label for this entity"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties as key-value pairs"
    )


class Relationship(BaseModel):
    """
    A relationship between two entities in the knowledge graph.

    Examples:
    - "Renesas" --[produces]--> "RH850 MCU"
    - "Factory fire" --[damaged]--> "N3 Building"
    - "Ford" --[uses]--> "RH850 MCU"
    """
    source_id: str = Field(
        description="The ID of the source entity"
    )
    relation_type: str = Field(
        description="The type of relationship (use snake_case)"
    )
    target_id: str = Field(
        description="The ID of the target entity"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Optional metadata about this relationship"
    )


class KnowledgeGraph(BaseModel):
    """
    A complete knowledge graph with entities and relationships.
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
# Knowledge Graph Extraction (from script 3)
# ============================================================================

KNOWLEDGE_GRAPH_EXTRACTION_PROMPT = """You are an expert in knowledge graph construction and information extraction.

Your task is to analyze a document and extract a knowledge graph with concrete entities and their relationships.

Guidelines for extraction:

1. **Entities**: Identify specific, concrete entities mentioned in the document.
   - Use clear, unique IDs (snake_case): e.g., 'renesas_company', 'n3_building', 'rh850_mcu'
   - Assign intuitive types: 'Organization', 'Location', 'Technology', 'Event', 'Product', etc.
   - Use the actual name from the document as the label
   - Add key properties mentioned in the text (dates, descriptions, attributes)

2. **Relationships**: Identify how entities connect to each other.
   - Use descriptive relation types (snake_case): e.g., 'produces', 'located_in', 'affected_by', 'supplies_to'
   - Only create relationships explicitly stated or strongly implied in the text
   - Ensure source_id and target_id match entity IDs exactly

3. **Keep it focused**:
   - Aim for 8-15 entities (enough to be interesting, not overwhelming)
   - Aim for 10-20 relationships (shows meaningful connections)
   - Focus on the most important concepts and their key relationships

4. **Be specific to the document**:
   - Extract real entities mentioned in the text, not generic placeholders
   - Use actual names, organizations, technologies, locations from the document
   - Capture domain-specific terminology

Return a structured knowledge graph (entities and relationships)."""


def extract_knowledge_graph_with_llm(
    text: str,
    document_name: str = "document",
) -> KnowledgeGraph:
    """
    Use an LLM to extract a knowledge graph from text.

    This function prompts an LLM to identify concrete entities and relationships
    in the text, then parses the response into a structured KnowledgeGraph object.

    Args:
        text: The input text to analyze
        document_name: Name of the document (used for progress messages)

    Returns:
        KnowledgeGraph: The extracted knowledge graph with entities and relationships
    """
    # Initialize the LLM with low temperature (deterministic output)
    # and bind structured output to get KnowledgeGraph directly
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
    print(f"    Extracting knowledge graph from {document_name}...")
    try:
        kg = structured_llm.invoke(messages)
        print(f"    ✓ Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
        return kg
    except Exception as e:
        print(f"\n    ERROR: Failed to extract knowledge graph: {e}")
        raise


# ============================================================================
# Knowledge Graph Merging
# ============================================================================

def merge_knowledge_graphs(graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
    """
    Merge multiple knowledge graphs into a single unified graph.

    This function:
    1. Combines all entities from all graphs
    2. Deduplicates entities with similar IDs or labels
    3. Combines all relationships
    4. Creates a unified graph that spans all source documents

    Args:
        graphs: List of KnowledgeGraph objects to merge

    Returns:
        KnowledgeGraph: A single merged graph containing all entities and relationships
    """
    if not graphs:
        return KnowledgeGraph(
            name="empty_graph",
            description="No graphs to merge",
            entities=[],
            relationships=[]
        )

    if len(graphs) == 1:
        return graphs[0]

    print(f"\n  Merging {len(graphs)} knowledge graphs...")

    # ========================================================================
    # Step 1: Merge entities
    # ========================================================================
    # Use a dict to deduplicate entities by ID
    # If two graphs have entities with the same ID, merge their properties
    merged_entities_dict: Dict[str, Entity] = {}

    for graph in graphs:
        for entity in graph.entities:
            if entity.id in merged_entities_dict:
                # Entity already exists - merge properties
                existing = merged_entities_dict[entity.id]
                # Update properties (new values override old ones)
                existing.properties.update(entity.properties)
            else:
                # New entity - add it
                merged_entities_dict[entity.id] = entity

    merged_entities = list(merged_entities_dict.values())

    # ========================================================================
    # Step 2: Merge relationships
    # ========================================================================
    # Use a set to deduplicate relationships (same source, relation, target)
    relationship_set: Set[Tuple[str, str, str]] = set()
    merged_relationships: List[Relationship] = []

    for graph in graphs:
        for rel in graph.relationships:
            # Create a unique key for this relationship
            rel_key = (rel.source_id, rel.relation_type, rel.target_id)

            if rel_key not in relationship_set:
                # Only keep relationships where both entities exist
                if (rel.source_id in merged_entities_dict and
                    rel.target_id in merged_entities_dict):
                    relationship_set.add(rel_key)
                    merged_relationships.append(rel)

    print(f"  ✓ Merged graph: {len(merged_entities)} entities, {len(merged_relationships)} relationships")

    return KnowledgeGraph(
        name="merged_knowledge_graph",
        description=f"Unified knowledge graph from {len(graphs)} source documents",
        entities=merged_entities,
        relationships=merged_relationships
    )


# ============================================================================
# Knowledge Graph Querying
# ============================================================================

class KnowledgeGraphQuerier:
    """
    A class for querying knowledge graphs to answer complex questions.

    This querier can:
    - Find entities by ID, label, or type
    - Traverse relationships between entities
    - Find paths connecting entities (multi-hop reasoning)
    - Answer questions using the LLM with graph context
    """

    def __init__(self, kg: KnowledgeGraph):
        """
        Initialize the querier with a knowledge graph.

        Args:
            kg: The knowledge graph to query
        """
        self.kg = kg

        # Build indexes for fast lookups
        self.entities_by_id: Dict[str, Entity] = {
            e.id: e for e in kg.entities
        }

        # Build adjacency list for relationship traversal
        # Format: {source_id: [(relation_type, target_id, relationship)]}
        self.outgoing: Dict[str, List[Tuple[str, str, Relationship]]] = {}
        self.incoming: Dict[str, List[Tuple[str, str, Relationship]]] = {}

        for rel in kg.relationships:
            # Outgoing edges (source -> target)
            if rel.source_id not in self.outgoing:
                self.outgoing[rel.source_id] = []
            self.outgoing[rel.source_id].append(
                (rel.relation_type, rel.target_id, rel)
            )

            # Incoming edges (target <- source)
            if rel.target_id not in self.incoming:
                self.incoming[rel.target_id] = []
            self.incoming[rel.target_id].append(
                (rel.relation_type, rel.source_id, rel)
            )

    def find_entities_by_label(self, label_substring: str) -> List[Entity]:
        """
        Find entities whose labels contain the given substring (case-insensitive).

        Args:
            label_substring: String to search for in entity labels

        Returns:
            List of matching entities
        """
        label_lower = label_substring.lower()
        return [
            e for e in self.kg.entities
            if label_lower in e.label.lower()
        ]

    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Find all entities of a given type.

        Args:
            entity_type: The entity type to search for (case-insensitive)

        Returns:
            List of matching entities
        """
        type_lower = entity_type.lower()
        return [
            e for e in self.kg.entities
            if e.type.lower() == type_lower
        ]

    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[Tuple[Entity, Relationship]]:
        """
        Get entities related to a given entity.

        Args:
            entity_id: The ID of the entity to start from
            relation_type: Optional - filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (related_entity, relationship) tuples
        """
        results = []

        # Get outgoing relationships (entity -> other)
        if direction in ["outgoing", "both"]:
            for rel_type, target_id, rel in self.outgoing.get(entity_id, []):
                if relation_type is None or rel_type == relation_type:
                    if target_id in self.entities_by_id:
                        results.append((self.entities_by_id[target_id], rel))

        # Get incoming relationships (other -> entity)
        if direction in ["incoming", "both"]:
            for rel_type, source_id, rel in self.incoming.get(entity_id, []):
                if relation_type is None or rel_type == relation_type:
                    if source_id in self.entities_by_id:
                        results.append((self.entities_by_id[source_id], rel))

        return results

    def find_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 4
    ) -> Optional[List[Tuple[Entity, Relationship]]]:
        """
        Find a path between two entities using breadth-first search.

        This enables multi-hop reasoning by traversing relationship chains.

        Args:
            start_entity_id: The ID of the starting entity
            end_entity_id: The ID of the target entity
            max_depth: Maximum path length to search

        Returns:
            List of (entity, relationship) tuples forming the path, or None if no path exists
        """
        if start_entity_id not in self.entities_by_id:
            return None
        if end_entity_id not in self.entities_by_id:
            return None
        if start_entity_id == end_entity_id:
            return []

        # BFS to find shortest path
        # Queue format: (current_entity_id, path_so_far)
        # path_so_far format: [(entity, relationship_to_next)]
        queue = [(start_entity_id, [])]
        visited = {start_entity_id}

        while queue:
            current_id, path = queue.pop(0)

            # Check depth limit
            if len(path) >= max_depth:
                continue

            # Get all outgoing relationships
            for rel_type, target_id, rel in self.outgoing.get(current_id, []):
                if target_id == end_entity_id:
                    # Found the target!
                    # Build the complete path
                    result = path + [(self.entities_by_id[target_id], rel)]
                    return result

                if target_id not in visited:
                    visited.add(target_id)
                    new_path = path + [(self.entities_by_id[target_id], rel)]
                    queue.append((target_id, new_path))

        # No path found
        return None

    def format_path(self, path: List[Tuple[Entity, Relationship]]) -> str:
        """
        Format a path into a human-readable string.

        Args:
            path: List of (entity, relationship) tuples

        Returns:
            Formatted string representation of the path
        """
        if not path:
            return "(empty path)"

        # Start with the first relationship's source
        if path[0][1].source_id in self.entities_by_id:
            start_entity = self.entities_by_id[path[0][1].source_id]
            result = f"{start_entity.label}"
        else:
            result = "?"

        # Add each step in the path
        for entity, rel in path:
            relation_label = rel.relation_type.replace("_", " ")
            result += f"\n  --[{relation_label}]-->\n{entity.label}"

        return result

    def answer_question_with_llm(self, question: str) -> str:
        """
        Use an LLM to answer a question using the knowledge graph as context.

        This is the key advantage over vector search:
        - The LLM can see the entire graph structure
        - It can reason about relationships and connections
        - It can traverse multi-hop paths to answer complex questions

        Args:
            question: The question to answer

        Returns:
            The LLM's answer based on the knowledge graph
        """
        # Build a text representation of the knowledge graph for the LLM
        kg_context = self._build_graph_context()

        # Create the prompt
        system_prompt = """You are an expert at reasoning over knowledge graphs to answer questions.

You have been given a knowledge graph containing entities and their relationships.
Use this structured information to answer the user's question.

Key capabilities:
- You can follow chains of relationships (multi-hop reasoning)
- You can connect information across different entities
- You can infer causal relationships from the graph structure
- You can identify temporal sequences from the relationships

Rules:
- Base your answer strictly on the entities and relationships in the knowledge graph
- If the graph doesn't contain enough information to fully answer the question, say so
- When answering, explain your reasoning by referencing the entities and relationships
- Show how you traverse the graph to reach your conclusion"""

        user_prompt = f"""Knowledge Graph:

{kg_context}

---

Question: {question}

Answer the question using the knowledge graph above. Explain your reasoning by showing which entities and relationships you used."""

        # Call the LLM
        llm = get_llm(temperature=0.0)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Handle potential list response format
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)

        return content if isinstance(content, str) else str(content)

    def _build_graph_context(self, max_entities: int = 100) -> str:
        """
        Build a text representation of the knowledge graph for LLM context.

        Args:
            max_entities: Maximum number of entities to include (to avoid context overflow)

        Returns:
            Formatted string representation of the graph
        """
        lines = []

        # List all entities
        lines.append("ENTITIES:")
        for i, entity in enumerate(self.kg.entities[:max_entities], 1):
            lines.append(f"\n{i}. [{entity.id}] {entity.label} (type: {entity.type})")
            if entity.properties:
                for key, value in entity.properties.items():
                    lines.append(f"   - {key}: {value}")

        if len(self.kg.entities) > max_entities:
            lines.append(f"\n... and {len(self.kg.entities) - max_entities} more entities")

        # List all relationships
        lines.append("\n\nRELATIONSHIPS:")
        for i, rel in enumerate(self.kg.relationships, 1):
            source_label = self.entities_by_id.get(rel.source_id, Entity(id="?", type="?", label="?")).label
            target_label = self.entities_by_id.get(rel.target_id, Entity(id="?", type="?", label="?")).label
            relation_label = rel.relation_type.replace("_", " ")

            lines.append(f"\n{i}. {source_label} --[{relation_label}]--> {target_label}")
            if rel.properties:
                lines.append(f"   (properties: {rel.properties})")

        return "\n".join(lines)


# ============================================================================
# Caching Functions
# ============================================================================

def save_knowledge_graph_to_cache(kg: KnowledgeGraph, pdf_path: Path) -> None:
    """
    Save a knowledge graph to cache to avoid re-extracting it.

    Args:
        kg: The knowledge graph to cache
        pdf_path: The source PDF path (used to generate cache filename)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use PDF filename as cache key
    cache_key = pdf_path.stem
    cache_path = CACHE_DIR / f"{cache_key}.json"

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(kg.model_dump(), f, indent=2, ensure_ascii=False)


def load_knowledge_graph_from_cache(pdf_path: Path) -> Optional[KnowledgeGraph]:
    """
    Load a knowledge graph from cache if it exists.

    Args:
        pdf_path: The source PDF path (used to generate cache filename)

    Returns:
        KnowledgeGraph if cached, None otherwise
    """
    cache_key = pdf_path.stem
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return KnowledgeGraph(**data)
    except Exception as e:
        print(f"  Warning: Failed to load cache for {pdf_path.name}: {e}")
        return None


# ============================================================================
# Main Functions
# ============================================================================

def build_unified_knowledge_graph(
    pdf_paths: List[Path],
    use_cache: bool = True
) -> KnowledgeGraph:
    """
    Build a unified knowledge graph from multiple PDFs.

    This function:
    1. Extracts text from each PDF
    2. Uses LLM to extract knowledge graphs from each
    3. Merges all graphs into a unified graph
    4. Caches individual graphs for faster subsequent runs

    Args:
        pdf_paths: List of PDF files to process
        use_cache: Whether to use cached knowledge graphs

    Returns:
        KnowledgeGraph: Unified knowledge graph from all PDFs
    """
    print(f"\n{'='*72}")
    print(f"  BUILDING UNIFIED KNOWLEDGE GRAPH")
    print(f"{'='*72}\n")
    print(f"  Processing {len(pdf_paths)} PDF files...\n")

    graphs = []

    for pdf_path in pdf_paths:
        print(f"  Processing: {pdf_path.name}")

        # Try to load from cache first
        if use_cache:
            cached_kg = load_knowledge_graph_from_cache(pdf_path)
            if cached_kg is not None:
                print(f"    ✓ Loaded from cache")
                graphs.append(cached_kg)
                continue

        # Extract text from PDF
        print(f"    Extracting text...")
        text = extract_text_from_pdf(pdf_path)
        print(f"    ✓ Extracted {len(text)} characters")

        # Extract knowledge graph using LLM
        kg = extract_knowledge_graph_with_llm(text, pdf_path.name)

        # Save to cache
        if use_cache:
            save_knowledge_graph_to_cache(kg, pdf_path)

        graphs.append(kg)
        print()

    # Merge all graphs
    unified_kg = merge_knowledge_graphs(graphs)

    return unified_kg


def main():
    """
    Main entry point for the knowledge graph querying script.
    """
    parser = argparse.ArgumentParser(
        description="Query knowledge graphs to answer complex questions"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to answer using the knowledge graph"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild knowledge graph cache from scratch"
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the unified knowledge graph structure"
    )
    args = parser.parse_args()

    # Use default question when none provided (no positional argument or empty)
    if not (args.question and args.question.strip()):
        args.question = DEFAULT_QUESTION

    # ========================================================================
    # Step 1: Load all PDFs from data directory
    # ========================================================================
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit(
            f"ERROR: No PDF files found in {DATA_DIR}\n"
            f"Add .pdf files to the data/ directory and run again."
        )

    print(f"\n{'='*72}")
    print(f"  KNOWLEDGE GRAPH QUERYING DEMO")
    print(f"{'='*72}\n")
    print(f"  Found {len(pdf_paths)} PDF files in {DATA_DIR}/")

    # ========================================================================
    # Step 2: Build unified knowledge graph
    # ========================================================================
    unified_kg = build_unified_knowledge_graph(
        pdf_paths,
        use_cache=not args.rebuild_cache
    )

    # ========================================================================
    # Step 3: Display graph structure (if requested)
    # ========================================================================
    if args.show_graph:
        print(f"\n{'─'*72}")
        print("  UNIFIED KNOWLEDGE GRAPH STRUCTURE")
        print(f"{'─'*72}\n")

        print(f"  Name: {unified_kg.name}")
        print(f"  Description: {unified_kg.description}")
        print(f"  Entities: {len(unified_kg.entities)}")
        print(f"  Relationships: {len(unified_kg.relationships)}\n")

        print("  Entity types:")
        entity_types = {}
        for entity in unified_kg.entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        for etype, count in sorted(entity_types.items()):
            print(f"    - {etype}: {count}")

        print("\n  Relationship types:")
        rel_types = {}
        for rel in unified_kg.relationships:
            rel_types[rel.relation_type] = rel_types.get(rel.relation_type, 0) + 1
        for rtype, count in sorted(rel_types.items()):
            print(f"    - {rtype}: {count}")

    # ========================================================================
    # Step 4: Initialize querier and answer the question
    # ========================================================================
    print(f"\n{'─'*72}")
    print("  ANSWERING QUESTION")
    print(f"{'─'*72}\n")

    querier = KnowledgeGraphQuerier(unified_kg)

    print(f"  Question: {args.question}\n")
    print("  Querying knowledge graph with LLM...\n")

    answer = querier.answer_question_with_llm(args.question)

    # Display the answer
    width = 72
    line = "═" * width
    thin = "─" * width

    print(line)
    print("  ANSWER FROM KNOWLEDGE GRAPH")
    print(thin)
    for paragraph in answer.strip().split("\n"):
        print(f"  {paragraph}")
    print(line)
    print()

    # ========================================================================
    # Step 5: Show comparison with vector search
    # ========================================================================
    print(f"{'─'*72}")
    print("  WHY THIS WORKS BETTER THAN VECTOR SEARCH")
    print(f"{'─'*72}\n")
    print("  Knowledge Graph Advantages:")
    print("  ✓ Can traverse relationships between entities (multi-hop reasoning)")
    print("  ✓ Links information across documents through shared entities")
    print("  ✓ Understands causal chains by following relationship sequences")
    print("  ✓ Handles temporal reasoning through time-based relationships")
    print("  ✓ Provides structured context to the LLM for better reasoning")
    print()
    print("  Vector Search Limitations (see 1_vector_search.py):")
    print("  ✗ Only finds similar text chunks, can't traverse connections")
    print("  ✗ Struggles with multi-hop questions spanning multiple documents")
    print("  ✗ Cannot infer causal relationships without explicit mentions")
    print("  ✗ Poor at temporal alignment and cross-document reasoning")
    print()
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
