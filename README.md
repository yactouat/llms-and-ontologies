# LLMs and ontologies

Demo scripts for getting started with LLMs and knowledge graphs/ontologies.

## what's in this repo

This repository contains five demonstration scripts:

1. **Vector Search (`1_vector_search.py`)** - Shows the limitations of vector search when dealing with complex, multi-hop reasoning or implicit relationships between documents
2. **Ontology Schema (`2_ontologies_owl.py`)** - Demonstrates how LLMs can extract ontology schemas (classes, properties, relationships) from documents and generate formal OWL ontologies for use in Protégé
3. **Knowledge Graph Instances (`3_ontologies_pydantic.py`)** - Shows how to extract concrete entities and their relationships to build a knowledge graph using Pydantic, with easy visualization options
4. **Knowledge Graph Querying (`4_query_ontology.py`)** - Demonstrates how to query across multiple knowledge graphs to answer complex questions that vector search cannot handle, showing the superiority of structured knowledge representation for multi-hop reasoning
5. **Neo4j Knowledge Graph Querying (`5_query_ontology_neo4j.py`)** - Shows how to store and query knowledge graphs in a Neo4j graph database for persistent storage, scalability, and advanced graph algorithms

These scripts illustrate why structured knowledge representation (ontologies and knowledge graphs) can complement vector search for knowledge-intensive tasks.

## get started

1. install [uv](https://docs.astral.sh/uv/) if you haven’t already.

2. from the project root, create a virtual environment and install dependencies:

   ```bash
   uv sync
   ```

   this creates a `.venv`, installs dependencies from `pyproject.toml`, and updates the lock file.

3. to run commands inside that environment:

   ```bash
   uv run <command>
   ```

   or activate the venv and use it as usual:

   ```bash
   source .venv/bin/activate  # linux/macos
   ```

4. **Set your Gemini API key**

   Copy the example env file and add your key:

   ```bash
   cp .env.example .env
   # Edit .env and set GOOGLE_API_KEY=your-actual-key
   ```

   Or export it in the shell:

   ```bash
   export GOOGLE_API_KEY="your-actual-key"
   ```

   Get an API key at [Google AI Studio](https://aistudio.google.com/apikey).

## demo scripts

### 1. vector search script (`1_vector_search.py`)

The script vectorizes all PDFs in the `data/` folder (using Gemini embeddings), persists the index in `chroma_db/`, and runs a vector search for a question.

**Usage:**

1. Add PDFs to the `data/` directory

2. Run the script:

   ```bash
   uv run python 1_vector_search.py
   ```

   Optional: pass a question and how many chunks to return:

   ```bash
   uv run python 1_vector_search.py "Your question here" -k 6
   ```

   The first run builds and saves the vector DB in `chroma_db/`. Later runs reuse it. To rebuild from scratch, remove the `chroma_db/` directory and run again.

### 2. ontology generation script (`2_ontologies_owl.py`)

The script uses an LLM to analyze a PDF document and automatically generate an OWL ontology (knowledge graph schema) that can be visualized in Protégé.

**What it does:**
- Extracts text from a PDF document
- Uses Gemini LLM to identify concepts (classes), properties, and relationships
- Generates an OWL file in RDF/XML format
- Creates a human-readable JSON summary

**Usage:**

1. Run the script with the default PDF (`cssanalyse295-en.pdf`):

   ```bash
   uv run python 2_ontologies_owl.py
   ```

   Or specify a different PDF:

   ```bash
   uv run python 2_ontologies_owl.py my_document.pdf
   ```

2. Optional arguments:

   ```bash
   # Specify output file name
   uv run python 2_ontologies_owl.py -o my_ontology.owl

   # Extract only specific pages (0-indexed)
   uv run python 2_ontologies_owl.py --start-page 0 --end-page 5
   ```

3. The script generates two files in the `ontologies/` directory:
   - `<name>.owl` - OWL ontology file (open in Protégé)
   - `<name>.json` - Human-readable summary

**Viewing the ontology:**

`sudo apt update && sudo apt install default-jre` if you don't have Java Runtime Environment

1. Download [Protégé](https://protege.stanford.edu/) (free ontology editor)
2. Open the generated `.owl` file in Protégé
3. Explore the tabs:
   - **Classes** - View the concept hierarchy
   - **Object Properties** - See relationships between entities
   - **Data Properties** - See attributes of entities
   - **OntoGraf** - Visualize the knowledge graph

### 3. knowledge graph extraction script (`3_ontologies_pydantic.py`)

The script uses an LLM to extract concrete entities and relationships from a PDF document, creating a knowledge graph that can be easily visualized and understood.

**What it does:**
- Extracts text from a PDF document
- Uses Gemini LLM to identify specific entities (people, organizations, technologies, events, etc.)
- Extracts relationships between entities
- Generates multiple output formats: JSON and DOT (Graphviz)
- Provides a human-readable text visualization

**Key difference from script 2:**
- Script 2 extracts **ontology schema** (what types of things exist and how they can relate)
- Script 3 extracts **instances** (actual entities and their specific relationships)

**Usage:**

1. Run the script with the default PDF (`cssanalyse295-en.pdf`):

   ```bash
   uv run python 3_ontologies_pydantic.py
   ```

   Or specify a different PDF:

   ```bash
   uv run python 3_ontologies_pydantic.py my_document.pdf
   ```

2. Optional arguments:

   ```bash
   # Specify output file name
   uv run python 3_ontologies_pydantic.py -o my_graph

   # Choose output format (json, dot, or both)
   uv run python 3_ontologies_pydantic.py --format json

   # Extract only specific pages (0-indexed)
   uv run python 3_ontologies_pydantic.py --start-page 0 --end-page 5
   ```

3. The script generates files in the `knowledge_graphs/` directory:
   - `<name>.json` - Structured JSON representation of entities and relationships
   - `<name>.dot` - Graphviz DOT file for visualization

**Visualizing the knowledge graph:**

Method 1: Using Graphviz (recommended for presentations)
```bash
sudo apt install graphviz  # if not already installed
dot -Tsvg knowledge_graphs/global_semiconductor_industry_knowledge_graph.dot -o tmp.svg
```

Method 2: Online visualization
1. Open http://www.webgraphviz.com/
2. Copy and paste the content of the `.dot` file
3. View the interactive graph visualization

Method 3: Use the text visualization printed to the console during execution

### 4. knowledge graph querying script (`4_query_ontology.py`)

The script demonstrates how to query across multiple knowledge graphs to answer complex questions that vector search fails at. It builds a unified knowledge graph from all PDFs in `data/` and uses it to answer multi-hop, causal, and temporal questions.

**What it does:**
- Loads all PDF files from the `data/` directory
- Extracts knowledge graphs from each using LLM (with caching for performance)
- Merges knowledge graphs into a unified graph spanning all documents
- Queries the graph using LLM with full structural context
- Demonstrates superiority over vector search for complex reasoning tasks

**Key advantages over vector search:**
- Can traverse relationships between entities (multi-hop reasoning)
- Links information across documents through shared entities
- Understands causal chains by following relationship sequences
- Handles temporal reasoning through time-based relationships
- Provides structured context to the LLM for better reasoning

**Usage:**

1. Run the script with a default question:

   ```bash
   uv run python 4_query_ontology.py
   ```

   Or ask a specific question:

   ```bash
   uv run python 4_query_ontology.py "How did the N3 Building fire affect Kentucky Truck Plant inventory?"
   ```

2. Optional arguments:

   ```bash
   # Display the unified knowledge graph structure before querying
   uv run python 4_query_ontology.py --show-graph "Your question here"

   # Rebuild the knowledge graph cache from scratch
   uv run python 4_query_ontology.py --rebuild-cache "Your question here"
   ```

3. The script automatically caches extracted knowledge graphs in `knowledge_graphs_cache/` for faster subsequent runs. Remove this directory to force re-extraction.

**Example questions that work well:**
- "How did the N3 Building fire affect Kentucky Truck Plant inventory?"
- "What specific equipment failure in Ibaraki Prefecture led to supplier-related production constraints?"
- "Why did the loss of 23 machines in a Japanese clean room cause a multi-billion dollar halt for F-150 production?"
- "Do the Renesas fire updates align with Ford's quarterly production decreases?"

**Why it works better than vector search:**
This script can answer the complex questions listed in the "vector search failing questions" section below because it understands the _relationships_ between entities across documents, not just textual similarity.

### 5. neo4j knowledge graph querying script (`5_query_ontology_neo4j.py`)

The script demonstrates how to use Neo4j graph database to store and query knowledge graphs for persistent storage, scalability, and advanced graph capabilities. It builds on script 4 but stores the knowledge graph in Neo4j instead of keeping it in memory.

**What it does:**
- Loads all PDF files from the `data/` directory
- Extracts knowledge graphs from each using LLM (with caching)
- Stores entities as nodes and relationships as edges in Neo4j
- Queries the graph using Cypher to get relevant context
- Uses LLM to interpret the graph context and answer questions

**Key advantages over in-memory graphs (script 4):**
- **Persistent storage**: graphs survive process restarts and can be reused
- **Scalability**: handles millions of entities and relationships efficiently
- **Advanced querying**: leverage Cypher's powerful graph query language
- **Graph algorithms**: use Neo4j's built-in analytics (shortest path, centrality, PageRank, etc.)
- **Multi-user access**: multiple processes can query the same graph concurrently
- **ACID transactions**: ensures data consistency
- **Visual exploration**: use Neo4j Browser to explore and visualize the graph interactively

**Prerequisites:**

1. Install Docker and docker-compose if not already installed
2. Start Neo4j database:

   ```bash
   docker-compose up -d
   ```

   This starts Neo4j on:
   - HTTP browser interface: http://localhost:7474
   - Bolt protocol (Python driver): bolt://localhost:7687
   - Default credentials: username=`neo4j`, password=`password123`

3. Install the neo4j Python driver (automatically done with `uv sync`):

   ```bash
   uv sync
   ```

**Usage:**

1. Start Neo4j (if not already running):

   ```bash
   docker-compose up -d
   ```

2. Run the script with a question:

   ```bash
   uv run python 5_query_ontology_neo4j.py
   ```

   Or ask a specific question:

   ```bash
   uv run python 5_query_ontology_neo4j.py "How did the N3 Building fire affect Kentucky Truck Plant inventory?"
   ```

3. Optional arguments:

   ```bash
   # Display Neo4j graph statistics before querying
   uv run python 5_query_ontology_neo4j.py --show-graph "Your question here"

   # Clear Neo4j and rebuild from scratch
   uv run python 5_query_ontology_neo4j.py --rebuild "Your question here"

   # Rebuild the cache from PDFs
   uv run python 5_query_ontology_neo4j.py --rebuild-cache "Your question here"
   ```

4. The script automatically:
   - Connects to Neo4j
   - Checks if the graph is already built
   - Builds the graph from PDFs if needed (using cache when available)
   - Queries Neo4j for relevant context
   - Uses LLM to answer the question

**Exploring the graph visually:**

1. Open Neo4j Browser: http://localhost:7474
2. Connect with:
   - Username: `neo4j`
   - Password: `password123`
3. Run Cypher queries to explore:

   ```cypher
   // View all entities (limit 50)
   MATCH (n:Entity) RETURN n LIMIT 50

   // View all relationships (limit 25)
   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

   // Find specific entities
   MATCH (n:Entity) WHERE n.label CONTAINS 'Renesas' RETURN n

   // Find paths between entities
   MATCH path = (a:Entity)-[*1..3]->(b:Entity)
   WHERE a.label CONTAINS 'fire' AND b.label CONTAINS 'Ford'
   RETURN path LIMIT 5
   ```

4. Click on nodes and relationships to explore the graph interactively

**Managing Neo4j:**

```bash
# Start Neo4j
docker-compose up -d

# Stop Neo4j
docker-compose down

# View logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Remove Neo4j and all data
docker-compose down -v
```

**Why use Neo4j over in-memory graphs:**
This script demonstrates production-ready knowledge graph storage. Use Neo4j when you need:
- Graphs that persist across runs
- Large-scale graphs (millions of nodes/edges)
- Advanced graph analytics and algorithms
- Multi-user concurrent access
- Integration with existing Neo4j infrastructure
- Visual graph exploration and debugging

### vector search failing questions (`1_vector_search.py`)

### 1. The "Ghost Relationship" Question
> **"How did the damage to the N3 Building clean room specifically impact the inventory levels at the Kentucky Truck Plant?"**

*   **Why Vector Search Fails:** The Renesas PDF mentions "N3 Building" and "clean room" but has zero mention of "Kentucky" or "Truck Plant." The Ford 10-K mentions the "Kentucky Truck Plant" but has zero mention of "N3." A vector search won't find the "RH850 chip" bridge between them.

### 2. The "Financial Root Cause" Question
> **"What specific equipment failure in Ibaraki Prefecture led to the 'supplier-related production constraints' cited in Ford's 2021 10-K?"**

*   **Why Vector Search Fails:** Ford's report uses vague corporate language ("supplier-related constraints"). The Renesas report is very specific about "plating equipment" and "overcurrent" in the Naka Factory (Ibaraki). The vector engine cannot link the "vague effect" in one document to the "specific cause" in the other without an ontological link.

### 3. The "Insignificant Component" Question
> **"Explain why the loss of 23 machines in a Japanese clean room caused a multi-billion dollar halt for the F-150 production line."**

*   **Why Vector Search Fails:** This requires a "Weight of Impact" understanding. The vector search will find the number "23" in the Renesas update and "multi-billion" in the Ford report, but it cannot explain the *causality* (the "One Dollar" bottleneck) unless the ontology explicitly defines that the *entire* truck depends on that *one* specific MCU family.

### 4. The "Timeline Sync" Question
> **"Do the 'Notice Regarding the Fire' updates from Renesas align with the specific quarterly production decreases reported by Ford in their 2021 filings?"**

*   **Why Vector Search Fails:** Vector search is notoriously bad at temporal alignment. It will pull snippets of dates from both, but won't be able to "join" them to show that the Q2 production dip at Ford matches the 100-day recovery lead time mentioned in Renesas's Update 3.

## when to use what

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **Vector Search** | Finding similar content, semantic search, initial document retrieval | Struggles with multi-hop reasoning, implicit relationships, temporal alignment, causal chains |
| **Ontologies** | Complex reasoning, relationship mapping, domain modeling, inference | Requires upfront schema design, can be rigid, needs expertise to build well |
| **Hybrid** | Best of both worlds - use vector search for retrieval, ontologies for reasoning | More complex to implement and maintain |

The scripts in this repo demonstrate both approaches so you can understand their strengths and weaknesses.
