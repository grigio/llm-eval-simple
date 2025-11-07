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

## Testing Commands
### Backend Testing
- **Run all backend tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov=. --cov-report=html`
- **Run specific test file**: `uv run pytest tests/unit/test_main.py`
- **Run tests with verbose output**: `uv run pytest -v`
- **Run only unit tests**: `uv run pytest -m unit`
- **Run only integration tests**: `uv run pytest -m integration`

### Frontend Testing
- **Run all frontend tests**: `cd frontend && npm run test`
- **Run tests once**: `cd frontend && npm run test:run`
- **Run tests with coverage**: `cd frontend && npm run test:coverage`
- **Run tests with UI**: `cd frontend && npm run test:ui`
- **Run tests in watch mode**: `cd frontend && npm run test:watch`

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
- **Language**: All code and comments must be in English