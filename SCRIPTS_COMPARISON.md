# Scripts Comparison Guide

Quick reference for choosing the right script for your demo needs.

## Overview

| Script | Purpose | Output | Best For |
|--------|---------|--------|----------|
| `1_vector_search.py` | Semantic similarity search | Retrieved text chunks + LLM answer | Finding similar content, baseline comparison |
| `2_ontologies_owl.py` | Extract ontology schema | OWL/RDF file for Protégé | Formal ontology modeling, schema definition |
| `3_ontologies_pydantic.py` | Extract knowledge graph instances | JSON + DOT visualization | Quick graph creation, presentations |
| `4_query_ontology.py` | Query in-memory graphs | Complex answers from merged graphs | Multi-hop reasoning, rapid prototyping |
| `5_query_ontology_neo4j.py` | Query persistent graphs in Neo4j | Same as 4, but with persistence | Production use, large graphs, visual exploration |

## When to Use Each Script

### Script 1: Vector Search (`1_vector_search.py`)
**Use when:**
- You need to find semantically similar text chunks
- You want to demonstrate vector search limitations
- You're doing initial document retrieval
- Your questions are simple and don't require multi-hop reasoning

**Don't use when:**
- Questions require connecting information across documents
- You need to follow relationship chains
- Temporal or causal reasoning is needed

### Script 2: Ontology Schema (`2_ontologies_owl.py`)
**Use when:**
- You need a formal ontology schema (OWL/RDF)
- You want to visualize class hierarchies in Protégé
- You're defining domain models and taxonomies
- You need formal reasoning with OWL reasoners
- You want to integrate with semantic web technologies

**Don't use when:**
- You need concrete instances (use script 3 instead)
- You want to query specific entities
- You need quick visualization for presentations

### Script 3: Knowledge Graph Instances (`3_ontologies_pydantic.py`)
**Use when:**
- You want to extract concrete entities from documents
- You need quick visualization (DOT/Graphviz)
- You're creating presentation materials
- You want a simple JSON representation of the graph
- You're prototyping a knowledge extraction pipeline

**Don't use when:**
- You need to query across multiple documents (use script 4/5)
- You need persistent storage (use script 5)
- You want to run graph algorithms

### Script 4: In-Memory Graph Querying (`4_query_ontology.py`)
**Use when:**
- You need to answer complex multi-hop questions
- You want to merge graphs from multiple documents
- You're doing rapid prototyping without infrastructure
- Your graph is small enough to fit in memory
- You want fast iteration during development

**Don't use when:**
- You need persistent storage between runs
- Your graph is too large for memory
- You want visual exploration tools
- Multiple processes need to access the graph
- You need advanced graph algorithms

### Script 5: Neo4j Graph Querying (`5_query_ontology_neo4j.py`)
**Use when:**
- You need persistent graph storage
- Your graph is large (millions of nodes/edges)
- You want to use Cypher for advanced queries
- You need visual exploration (Neo4j Browser)
- Multiple processes need concurrent access
- You want to use graph algorithms (shortest path, centrality, etc.)
- You're building a production system

**Don't use when:**
- You're doing quick prototypes (use script 4 instead)
- You don't want to manage Docker/infrastructure
- Your graph is small and temporary
- You don't need the extra features Neo4j provides

## Progression Path

For a complete demo, show scripts in this order:

1. **Start with Script 1** - Show vector search baseline
2. **Show Script 2 or 3** - Introduce knowledge graphs (choose based on audience)
   - Script 2: For formal ontology/semantic web audience
   - Script 3: For general tech audience
3. **Use Script 4** - Demonstrate graph querying advantages
4. **End with Script 5** - Show production-ready solution with Neo4j

## Key Differences: Script 4 vs Script 5

| Feature | Script 4 (In-Memory) | Script 5 (Neo4j) |
|---------|---------------------|------------------|
| **Storage** | RAM (temporary) | Neo4j database (persistent) |
| **Scalability** | Limited by memory | Millions of nodes |
| **Query Language** | Python code | Cypher |
| **Visualization** | Text output only | Neo4j Browser (interactive) |
| **Graph Algorithms** | Manual implementation | Built-in (shortest path, etc.) |
| **Multi-user** | Single process | Concurrent access |
| **Setup** | None | Docker + Neo4j |
| **Speed** | Very fast (in RAM) | Fast (with indexes) |
| **Best For** | Prototyping | Production |

## Common Questions

**Q: Which script should I use for my talk about LLMs and knowledge graphs?**
A: Use scripts 1, 3, and 4 for a 20-minute talk. Add script 5 if discussing production deployment.

**Q: I want to show vector search failures. Which scripts?**
A: Compare script 1 (vector search) with script 4 or 5 (graph querying) using the same questions.

**Q: How do I visualize the knowledge graph?**
A: Script 3 for static images (Graphviz), Script 5 for interactive exploration (Neo4j Browser).

**Q: Can I use real data instead of PDFs?**
A: Yes! All scripts use the `utils.py` functions. Modify `extract_text_from_pdf()` to load from your data source.

**Q: Which script is fastest?**
A: Script 4 (in-memory) is fastest. Script 5 requires database queries but scales better.

## Tips for Demos

1. **Always start with vector search failures** to motivate why knowledge graphs are needed
2. **Use the same questions** across scripts to show clear improvements
3. **Keep your PDFs small** (2-3 documents) to keep extraction time reasonable during live demos
4. **Pre-build the Neo4j graph** before presenting script 5 (use `--rebuild` beforehand)
5. **Show Neo4j Browser** visually during the script 5 demo - it's impressive!

## Example Demo Flow (30 minutes)

```bash
# 1. Show vector search limitation (5 min)
uv run python 1_vector_search.py "How did the N3 Building fire affect Kentucky Truck Plant inventory?"

# 2. Extract knowledge graph (5 min)
uv run python 3_ontologies_pydantic.py
# Show the generated visualization

# 3. Query the graph (10 min)
uv run python 4_query_ontology.py "How did the N3 Building fire affect Kentucky Truck Plant inventory?"
# Compare answer with script 1

# 4. Show Neo4j persistence and visualization (10 min)
docker-compose up -d
uv run python 5_query_ontology_neo4j.py --rebuild --show-graph
# Open Neo4j Browser and explore visually
# Run sample Cypher queries
```

## Environment Variables

```bash
# Required for all scripts
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional for script 5 (defaults work with docker-compose.yml)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"
```
