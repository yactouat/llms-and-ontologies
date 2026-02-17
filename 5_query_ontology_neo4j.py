#!/usr/bin/env python3
"""
Demo: Query knowledge graphs using Neo4j graph database.

This script demonstrates how to:
1. Load multiple PDF documents from data/
2. Extract knowledge graphs from each using LLM
3. Store knowledge graphs in a Neo4j graph database
4. Query the graph using Cypher to answer complex questions
5. Use LLM to interpret query results and generate answers

Key advantages over in-memory graphs (4_query_ontology.py):
- Persistent storage: graphs survive process restarts
- Scalability: handle millions of entities and relationships
- Advanced querying: leverage Cypher for powerful graph queries
- Graph algorithms: use Neo4j's built-in graph analytics
- Multi-user access: multiple processes can query the same graph
- ACID transactions: ensure data consistency

Prerequisites:
- Docker and docker-compose installed
- Neo4j running via: docker-compose up -d

Requires GOOGLE_API_KEY in the environment.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

# Neo4j Python driver for connecting to the database
from neo4j import GraphDatabase, Driver

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils import get_llm, extract_text_from_pdf, load_pdfs_from_directory


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
CACHE_DIR = Path("knowledge_graphs_cache")
DEFAULT_QUESTION = "How did the N3 Building fire affect Kentucky Truck Plant inventory?"

# Neo4j connection settings
# These match the docker-compose.yml configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")


# ============================================================================
# Pydantic Models (same as 4_query_ontology.py)
# ============================================================================
# These models represent entities and relationships in knowledge graphs.
# They are used both for extraction and for storing in Neo4j.


class Entity(BaseModel):
    """
    A concrete entity (instance) in the knowledge graph.

    In Neo4j, entities become nodes with:
    - A unique ID (becomes node property)
    - A type label (becomes Neo4j label)
    - Properties stored as node properties
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

    In Neo4j, relationships become edges with:
    - Source and target nodes (identified by IDs)
    - A relationship type (becomes Neo4j relationship type)
    - Properties stored as relationship properties
    """
    source_id: str = Field(
        description="The ID of the source entity"
    )
    relation_type: str = Field(
        description="The type of relationship (use UPPER_SNAKE_CASE for Neo4j convention)"
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

    This will be stored in Neo4j as a set of nodes and edges.
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
# Knowledge Graph Extraction (from script 4)
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
   - Use descriptive relation types (UPPER_SNAKE_CASE for Neo4j): e.g., 'PRODUCES', 'LOCATED_IN', 'AFFECTED_BY', 'SUPPLIES_TO'
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
---

IMPORTANT: Use UPPER_SNAKE_CASE for relationship types (e.g., PRODUCES, LOCATED_IN) to follow Neo4j conventions.""")
    ]

    # Invoke the LLM; with_structured_output returns a KnowledgeGraph directly
    print(f"    Extracting knowledge graph from {document_name}...")
    try:
        kg = structured_llm.invoke(messages)
        print(f"    ✓ Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")

        # Normalize relationship types to UPPER_SNAKE_CASE for Neo4j
        for rel in kg.relationships:
            rel.relation_type = rel.relation_type.upper().replace(" ", "_")

        return kg
    except Exception as e:
        print(f"\n    ERROR: Failed to extract knowledge graph: {e}")
        raise


# ============================================================================
# Caching Functions (same as script 4)
# ============================================================================

def save_knowledge_graph_to_cache(kg: KnowledgeGraph, pdf_path: Path) -> None:
    """
    Save a knowledge graph to cache to avoid re-extracting it.

    Args:
        kg: The knowledge graph to cache
        pdf_path: The source PDF path (used to generate cache filename)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
# Neo4j Connection and Management
# ============================================================================

class Neo4jKnowledgeGraph:
    """
    A class for managing knowledge graphs in Neo4j.

    This class handles:
    - Connecting to Neo4j database
    - Creating/updating nodes and relationships
    - Querying the graph with Cypher
    - Clearing the database
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j database.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        try:
            # Create a driver instance - this is the connection pool
            self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

            # Test the connection
            self.driver.verify_connectivity()
            print(f"  ✓ Connected to Neo4j at {uri}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j at {uri}.\n"
                f"Error: {e}\n\n"
                f"Make sure Neo4j is running:\n"
                f"  docker-compose up -d\n\n"
                f"Check connection settings in the script or set environment variables:\n"
                f"  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
            )

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            print("  ✓ Closed Neo4j connection")

    def clear_database(self):
        """
        Clear all nodes and relationships from the database.

        WARNING: This deletes ALL data in the database!
        Use only for demos and testing.
        """
        with self.driver.session() as session:
            # Delete all relationships first, then all nodes
            # (Neo4j requires relationships to be deleted before nodes)
            result = session.run("MATCH (n) DETACH DELETE n")
            print("  ✓ Cleared Neo4j database")

    def create_indexes(self):
        """
        Create indexes for better query performance.

        Indexes speed up lookups by entity ID and type.
        These are especially important for large graphs.
        """
        with self.driver.session() as session:
            # Create index on entity ID for fast lookups
            # This makes MATCH (e {entity_id: "..."}) queries very fast
            try:
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)")
                print("  ✓ Created index on entity_id")
            except Exception as e:
                print(f"  Note: Index creation skipped: {e}")

    def store_knowledge_graph(self, kg: KnowledgeGraph):
        """
        Store a knowledge graph in Neo4j.

        This function:
        1. Creates a node for each entity with its properties
        2. Creates a relationship for each edge connecting entities

        Args:
            kg: The knowledge graph to store
        """
        with self.driver.session() as session:
            # ================================================================
            # Step 1: Create all entity nodes
            # ================================================================
            # Use MERGE instead of CREATE to avoid duplicates
            # MERGE will update existing nodes or create new ones
            for entity in kg.entities:
                # Build the properties dict for the node
                # We store all entity properties plus the entity_id and label
                node_props = {
                    "entity_id": entity.id,
                    "label": entity.label,
                    **entity.properties  # Merge in additional properties
                }

                # Cypher query to create/update the node
                # The node gets:
                # - A generic "Entity" label (for queries like MATCH (e:Entity))
                # - A specific type label (for queries like MATCH (e:Organization))
                # - All properties from node_props
                query = f"""
                MERGE (e:Entity:{entity.type} {{entity_id: $entity_id}})
                SET e += $props
                """

                session.run(query, entity_id=entity.id, props=node_props)

            print(f"    ✓ Stored {len(kg.entities)} entities as nodes")

            # ================================================================
            # Step 2: Create all relationships
            # ================================================================
            # Connect entities with typed relationships
            for rel in kg.relationships:
                # Cypher query to create a relationship
                # Note: We match entities by entity_id, then create the relationship
                # The relationship type is dynamic (uses the relation_type from the data)
                query = f"""
                MATCH (source:Entity {{entity_id: $source_id}})
                MATCH (target:Entity {{entity_id: $target_id}})
                MERGE (source)-[r:{rel.relation_type}]->(target)
                SET r += $props
                """

                session.run(
                    query,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    props=rel.properties
                )

            print(f"    ✓ Stored {len(kg.relationships)} relationships as edges")

    def get_graph_summary(self) -> Dict:
        """
        Get a summary of what's in the Neo4j database.

        Returns:
            Dict with counts of nodes, relationships, node types, and relationship types
        """
        with self.driver.session() as session:
            # Count total nodes
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = result.single()["count"]

            # Count total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]

            # Get node type distribution
            result = session.run("""
                MATCH (n:Entity)
                RETURN labels(n) as labels, count(*) as count
                ORDER BY count DESC
            """)
            node_types = {}
            for record in result:
                # labels() returns a list like ['Entity', 'Organization']
                # We want the specific type (not 'Entity')
                labels = [l for l in record["labels"] if l != "Entity"]
                if labels:
                    node_types[labels[0]] = record["count"]

            # Get relationship type distribution
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            rel_types = {record["type"]: record["count"] for record in result}

            return {
                "nodes": node_count,
                "relationships": rel_count,
                "node_types": node_types,
                "relationship_types": rel_types
            }

    def query_for_question_context(self, question: str, max_hops: int = 3) -> str:
        """
        Query Neo4j to get relevant context for answering a question.

        This function:
        1. Extracts key terms from the question
        2. Finds entities matching those terms
        3. Retrieves their neighborhoods (connected entities)
        4. Formats the results as text for the LLM

        Args:
            question: The question to find context for
            max_hops: Maximum relationship hops to traverse

        Returns:
            Formatted text representation of relevant graph context
        """
        with self.driver.session() as session:
            # ============================================================
            # Strategy: Find entities mentioned in the question,
            # then get their neighborhoods (connected entities)
            # ============================================================

            # Extract potential entity labels from the question
            # In a production system, you'd use NER or the LLM for this
            # For demo, we'll just search for entities whose labels match words in the question
            question_words = [w.lower() for w in question.split() if len(w) > 3]

            context_parts = []

            # Find entities whose labels contain question words
            for word in question_words[:5]:  # Limit to first 5 significant words
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.label) CONTAINS $word
                RETURN e.entity_id as id, e.label as label, labels(e) as types, properties(e) as props
                LIMIT 3
                """

                result = session.run(query, word=word)

                for record in result:
                    entity_id = record["id"]
                    entity_label = record["label"]
                    entity_types = [t for t in record["types"] if t != "Entity"]

                    context_parts.append(f"\n=== Entity: {entity_label} ===")
                    context_parts.append(f"ID: {entity_id}")
                    context_parts.append(f"Type: {', '.join(entity_types)}")

                    # Get entity properties
                    props = record["props"]
                    if props:
                        context_parts.append("Properties:")
                        for key, value in props.items():
                            if key not in ["entity_id", "label"]:
                                context_parts.append(f"  - {key}: {value}")

                    # Get outgoing relationships
                    rel_query = f"""
                    MATCH (e:Entity {{entity_id: $entity_id}})-[r]->(target:Entity)
                    RETURN type(r) as rel_type, target.entity_id as target_id, target.label as target_label
                    LIMIT 10
                    """

                    rel_result = session.run(rel_query, entity_id=entity_id)
                    relationships = list(rel_result)

                    if relationships:
                        context_parts.append("\nRelationships:")
                        for rel_record in relationships:
                            rel_type = rel_record["rel_type"].replace("_", " ")
                            target_label = rel_record["target_label"]
                            context_parts.append(f"  → {rel_type} → {target_label}")

                    # Get incoming relationships
                    incoming_query = f"""
                    MATCH (source:Entity)-[r]->(e:Entity {{entity_id: $entity_id}})
                    RETURN type(r) as rel_type, source.entity_id as source_id, source.label as source_label
                    LIMIT 10
                    """

                    incoming_result = session.run(incoming_query, entity_id=entity_id)
                    incoming_rels = list(incoming_result)

                    if incoming_rels:
                        for rel_record in incoming_rels:
                            rel_type = rel_record["rel_type"].replace("_", " ")
                            source_label = rel_record["source_label"]
                            context_parts.append(f"  ← {rel_type} ← {source_label}")

            # If we didn't find specific entities, provide a general graph overview
            if not context_parts:
                context_parts.append("\n=== General Graph Overview ===")

                # Get a sample of entities and relationships
                overview_query = """
                MATCH (e:Entity)
                RETURN e.entity_id as id, e.label as label, labels(e) as types
                LIMIT 20
                """
                result = session.run(overview_query)

                context_parts.append("\nSample Entities:")
                for record in result:
                    types = [t for t in record["types"] if t != "Entity"]
                    context_parts.append(f"  - {record['label']} ({', '.join(types)})")

                # Get sample relationships
                rel_overview_query = """
                MATCH (source:Entity)-[r]->(target:Entity)
                RETURN source.label as source, type(r) as rel_type, target.label as target
                LIMIT 15
                """
                result = session.run(rel_overview_query)

                context_parts.append("\nSample Relationships:")
                for record in result:
                    rel_type = record["rel_type"].replace("_", " ")
                    context_parts.append(f"  - {record['source']} → {rel_type} → {record['target']}")

            return "\n".join(context_parts)


# ============================================================================
# Question Answering with LLM
# ============================================================================

def answer_question_with_neo4j(
    neo4j_kg: Neo4jKnowledgeGraph,
    question: str
) -> str:
    """
    Use an LLM to answer a question using Neo4j graph context.

    This is the key integration:
    1. Query Neo4j to get relevant entities and relationships
    2. Format the graph data as text
    3. Pass it to the LLM as context
    4. Get a natural language answer

    Args:
        neo4j_kg: The Neo4j knowledge graph connection
        question: The question to answer

    Returns:
        The LLM's answer based on the knowledge graph
    """
    # Get relevant context from Neo4j
    print("  Querying Neo4j for relevant context...")
    graph_context = neo4j_kg.query_for_question_context(question)
    print("  ✓ Retrieved graph context\n")

    # Create the prompt for the LLM
    system_prompt = """You are an expert at reasoning over knowledge graphs to answer questions.

You have been given context from a Neo4j knowledge graph containing entities and their relationships.
Use this structured information to answer the user's question.

Key capabilities:
- You can follow chains of relationships (multi-hop reasoning)
- You can connect information across different entities
- You can infer causal relationships from the graph structure
- You can identify temporal sequences from the relationships

Rules:
- Base your answer strictly on the entities and relationships in the graph context
- If the graph doesn't contain enough information to fully answer the question, say so
- When answering, explain your reasoning by referencing the entities and relationships
- Show how you traverse the graph to reach your conclusion"""

    user_prompt = f"""Graph Context from Neo4j:

{graph_context}

---

Question: {question}

Answer the question using the knowledge graph context above. Explain your reasoning by showing which entities and relationships you used."""

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


# ============================================================================
# Main Functions
# ============================================================================

def build_neo4j_knowledge_graph(
    neo4j_kg: Neo4jKnowledgeGraph,
    pdf_paths: List[Path],
    use_cache: bool = True
) -> None:
    """
    Build a knowledge graph in Neo4j from multiple PDFs.

    This function:
    1. Extracts text from each PDF
    2. Uses LLM to extract knowledge graphs from each
    3. Stores all graphs in Neo4j
    4. Caches individual graphs for faster subsequent runs

    Args:
        neo4j_kg: Neo4j connection
        pdf_paths: List of PDF files to process
        use_cache: Whether to use cached knowledge graphs
    """
    print(f"\n{'='*72}")
    print(f"  BUILDING KNOWLEDGE GRAPH IN NEO4J")
    print(f"{'='*72}\n")
    print(f"  Processing {len(pdf_paths)} PDF files...\n")

    for pdf_path in pdf_paths:
        print(f"  Processing: {pdf_path.name}")

        # Try to load from cache first
        if use_cache:
            cached_kg = load_knowledge_graph_from_cache(pdf_path)
            if cached_kg is not None:
                print(f"    ✓ Loaded from cache")
                neo4j_kg.store_knowledge_graph(cached_kg)
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

        # Store in Neo4j
        neo4j_kg.store_knowledge_graph(kg)
        print()


def main():
    """
    Main entry point for the Neo4j knowledge graph querying script.
    """
    parser = argparse.ArgumentParser(
        description="Query knowledge graphs in Neo4j to answer complex questions"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to answer using the knowledge graph"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear Neo4j and rebuild knowledge graph from scratch"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild knowledge graph cache from PDFs"
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the Neo4j knowledge graph statistics"
    )
    args = parser.parse_args()

    # Use default question when none provided
    if not (args.question and args.question.strip()):
        args.question = DEFAULT_QUESTION

    # ========================================================================
    # Step 1: Connect to Neo4j
    # ========================================================================
    print(f"\n{'='*72}")
    print(f"  NEO4J KNOWLEDGE GRAPH QUERYING DEMO")
    print(f"{'='*72}\n")

    try:
        neo4j_kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except ConnectionError as e:
        print(f"\n{e}\n")
        return

    try:
        # ====================================================================
        # Step 2: Clear database if requested
        # ====================================================================
        if args.rebuild:
            print("\n  Clearing Neo4j database...")
            neo4j_kg.clear_database()
            print()

        # ====================================================================
        # Step 3: Check if we need to build the graph
        # ====================================================================
        summary = neo4j_kg.get_graph_summary()

        if summary["nodes"] == 0 or args.rebuild:
            # Load PDFs and build graph
            pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

            if not pdf_paths:
                raise SystemExit(
                    f"ERROR: No PDF files found in {DATA_DIR}\n"
                    f"Add .pdf files to the data/ directory and run again."
                )

            print(f"  Found {len(pdf_paths)} PDF files in {DATA_DIR}/")

            # Create indexes for better performance
            neo4j_kg.create_indexes()

            # Build the graph
            build_neo4j_knowledge_graph(
                neo4j_kg,
                pdf_paths,
                use_cache=not args.rebuild_cache
            )

            # Get updated summary
            summary = neo4j_kg.get_graph_summary()

        # ====================================================================
        # Step 4: Display graph statistics (if requested or after build)
        # ====================================================================
        if args.show_graph or args.rebuild:
            print(f"\n{'─'*72}")
            print("  NEO4J KNOWLEDGE GRAPH STATISTICS")
            print(f"{'─'*72}\n")

            print(f"  Total Entities: {summary['nodes']}")
            print(f"  Total Relationships: {summary['relationships']}\n")

            print("  Entity Types:")
            for etype, count in sorted(summary['node_types'].items(), key=lambda x: -x[1]):
                print(f"    - {etype}: {count}")

            print("\n  Relationship Types:")
            for rtype, count in sorted(summary['relationship_types'].items(), key=lambda x: -x[1]):
                rel_label = rtype.replace("_", " ")
                print(f"    - {rel_label}: {count}")

        # ====================================================================
        # Step 5: Answer the question
        # ====================================================================
        print(f"\n{'─'*72}")
        print("  ANSWERING QUESTION")
        print(f"{'─'*72}\n")

        print(f"  Question: {args.question}\n")

        answer = answer_question_with_neo4j(neo4j_kg, args.question)

        # Display the answer
        width = 72
        line = "═" * width
        thin = "─" * width

        print(line)
        print("  ANSWER FROM NEO4J KNOWLEDGE GRAPH")
        print(thin)
        for paragraph in answer.strip().split("\n"):
            print(f"  {paragraph}")
        print(line)
        print()

        # ====================================================================
        # Step 6: Show advantages
        # ====================================================================
        print(f"{'─'*72}")
        print("  WHY NEO4J WORKS BETTER THAN IN-MEMORY GRAPHS")
        print(f"{'─'*72}\n")
        print("  Neo4j Advantages:")
        print("  ✓ Persistent storage: graphs survive process restarts")
        print("  ✓ Scalability: handles millions of entities efficiently")
        print("  ✓ Cypher queries: powerful declarative graph query language")
        print("  ✓ Graph algorithms: built-in analytics (shortest path, centrality, etc.)")
        print("  ✓ Multi-user: concurrent access from multiple processes")
        print("  ✓ ACID transactions: ensures data consistency")
        print("  ✓ Visualization: use Neo4j Browser to explore the graph visually")
        print()
        print("  To explore the graph visually:")
        print(f"  1. Open http://localhost:7474 in your browser")
        print(f"  2. Connect with username: {NEO4J_USER}, password: {NEO4J_PASSWORD}")
        print(f"  3. Run Cypher queries like:")
        print(f"     MATCH (n) RETURN n LIMIT 50")
        print(f"     MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")
        print()
        print(f"{'='*72}\n")

    finally:
        # Always close the Neo4j connection
        neo4j_kg.close()


if __name__ == "__main__":
    main()
