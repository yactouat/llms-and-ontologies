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
