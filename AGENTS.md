# AGENTS.md

## Commands
- **Run main script**: `uv run python main.py`
- **Run specific actions**: `uv run python main.py --actions answer,evaluate,serve --pattern "prompts/*"`
- **Start web server**: `uv run python main.py --actions serve`
- **Start API server**: `uv run python api_server.py` (runs on port 4000)
- **Start web UI**: `cd frontend && npm run dev` (runs on port 3000)
- **Install dependencies**: `uv sync`
- **Add dependency**: `uv add <package>`
- **Remove dependency**: `uv remove <package>`

## Code Style
- **Package Manager**: Use `uv` (not pip)
- **Execution**: Use `uv run` for all Python commands
- **Dependencies**: Define in `pyproject.toml`, use `uv.lock` for reproducibility
- **Python Version**: Requires Python 3.13+
- **Structure**: Use dataclasses for config, separate functions clearly
- **Error Handling**: Use try/except for API requests, return None/False on failure
- **Type Hints**: Use typing module (List, Dict, Any)
- **Imports**: Standard library first, then third-party
- **File Encoding**: Always UTF-8
- **Constants**: Define at module level in UPPER_CASE
- **Virtual Environment**: `uv` manages automatically - no manual venv needed
- **Python Execution**: Always use `uv run python`