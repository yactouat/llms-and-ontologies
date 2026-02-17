# LLMs and ontologies

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

## running the vector search script (`1_vector_search.py`)

The script vectorizes all PDFs in the `data/` folder (using Gemini embeddings), persists the index in `chroma_db/`, and runs a vector search for a question.

1. **Add PDFs to index**

   Put one or more `.pdf` files in the `data/` directory.

2. **Run the script**

   ```bash
   uv run python 1_vector_search.py
   ```

   Optional: pass a question and how many chunks to return:

   ```bash
   uv run python 1_vector_search.py "Your question here" -k 6
   ```

   The first run builds and saves the vector DB in `chroma_db/`. Later runs reuse it. To rebuild from scratch, remove the `chroma_db/` directory and run again.

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
