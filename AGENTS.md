# AGENTS.md

## Build/Lint/Test Commands
- **Run main script**: `uv run python main.py`
- **Run with specific actions**: `uv run python main.py --actions answer,evaluate,serve --pattern "prompts/*"`
- **Start web server**: `uv run python main.py --actions serve`
- **Install dependencies**: `uv sync`
- **Add dependency**: `uv add <package>`
- **Remove dependency**: `uv remove <package>`

## Code Style Guidelines
- **Package Manager**: Use `uv` for all Python package management (not pip)
- **Execution**: Use `uv run` to execute scripts within the project environment
- **Dependencies**: Define in `pyproject.toml`, rely on `uv.lock` for reproducibility
- **Python Version**: Requires Python 3.13+
- **Code Structure**: Follow existing patterns - dataclasses for config, clear function separation
- **Error Handling**: Use try/except blocks for API requests, return None/False on failure
- **Type Hints**: Use typing module (List, Dict, Any) for function signatures
- **Imports**: Group standard library imports first, then third-party imports
- **File Encoding**: Always use UTF-8 encoding for file operations
- **Constants**: Define file paths and constants at module level in UPPER_CASE
- **Virtual Environment**: `uv` automatically manages a virtual environment. There's no need to manually create or activate one with `python -m venv` or `source venv/bin/activate`. Use `uv sync` to install dependencies from the `pyproject.toml` and `uv.lock` files, creating or updating the virtual environment as needed.
- **Python via uv**: Always use `uv run python` to execute Python commands. Example: `uv run python --version` returns Python 3.13.1