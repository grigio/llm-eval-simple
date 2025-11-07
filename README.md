# LLM Eval Simple

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![benchmark report](./static/benchmark-report.png)

A simple tool for evaluating Large Language Models (LLMs) using a set of prompts and expected answers. It supports testing multiple models via an OpenAI-compatible API endpoint, measures response times, evaluates correctness (using an optional evaluator model or exact matching), and generates a summary report in tabular format.

This script is useful for benchmarking LLM performance on custom datasets, such as accuracy on specific tasks or questions.

## Features

- **Batch Testing**: Evaluate multiple LLM models simultaneously
- **Flexible Evaluation**: Use AI evaluator models or exact string matching
- **Performance Metrics**: Track response times and accuracy
- **Rich Reporting**: Detailed tables and summary statistics
- **Web Dashboard**: Interactive visualization of results
- **Configurable**: Environment-based configuration for different setups

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env

# Edit .env with your API endpoint and model names
# Then run evaluation
uv run python main.py

# Start web dashboard
./start-dashboard.sh
```

## Project Structure

```
llm-eval-simple/
├── main.py                 # Main evaluation script
├── api_server.py          # REST API server
├── server.py               # Web server for dashboard
├── api_client.py           # OpenAI-compatible API client
├── reporting.py            # Result formatting and display
├── validation.py           # Input validation utilities
├── file_utils.py           # File operations and utilities
├── shared.py               # Shared constants and utilities
├── prompts/                # Input prompt files
├── answers/                # Expected answer files
├── frontend/               # React dashboard application
├── tests/                  # Backend test suite
└── static/                 # Static assets
```

## Prerequisites

- Python 3.13+.
- [uv](https://github.com/astral-sh/uv) installed for dependency management (a fast alternative to pip and venv).
- Access to an OpenAI-compatible API endpoint (e.g., local server or hosted service) for model inference.
- Directories for prompts and answers (created automatically if missing).

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd llm-eval-simple
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required packages from `pyproject.toml` or `requirements.txt`.

   If you prefer not to use uv, you can manually install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix-based systems
   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

   Note: The script assumes uv for running, but you can adapt it for standard Python.

## Configuration

1. Create a `.env` file in the root directory based on `.env.example`:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your settings:
   - `ENDPOINT_URL`: Your OpenAI-compatible API endpoint (default: `http://localhost:9292/v1/chat/completions`).
   - `API_KEY`: Your API key for authentication with the OpenAI-compatible API (optional).
   - `MODEL_NAMES`: Comma-separated list of model names to test (e.g., `gemma-3-270m-it-Q4_K_M,Qwen3-8B-Q4_K_M`).
   - `MODEL_EVALUATOR`: Optional model name for evaluating correctness (if empty, uses exact matching).

2. Prepare your test data:
   - Place prompt files in the `prompts` directory (e.g., `1-math-question.txt`).
   - Place corresponding expected answer files in the `answers` directory with matching names (e.g., `1-math-question.txt`).
   - Files should contain plain text: prompts for input to the model, answers for comparison.
   - Use consistent naming and ensure files are UTF-8 encoded.

## Usage

### Running Evaluations

Run the evaluation script with customizable actions:

```bash
# Run all actions (answer, evaluate, render, serve)
uv run python main.py

# Run specific actions with pattern filtering
uv run python main.py --actions answer,evaluate,serve --pattern "prompts/REASON*"

# Available actions:
# - answer: Generate model responses for prompts
# - evaluate: Evaluate correctness of responses
# - render: Display results in terminal
# - serve: Start web dashboard
```

### Starting the Dashboard

For a web-based dashboard to view results, use the `start-dashboard.sh` script:

```bash
./start-dashboard.sh
```

This will start:
- API server on port 4000
- Web UI on port 3000

Alternatively, start components manually:
```bash
# Start API server
uv run python api_server.py

# Start web UI (in another terminal)
cd frontend && npm run dev
```

- This will process all prompt files, test each model, evaluate results, and print detailed per-file results followed by a summary table.
- Output includes:
  - Per-model testing logs.
  - Detailed table with model, file, correctness, and response time.
  - Summary table with accuracy percentage and average response time.

Example output snippet:

```text
...
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gemma-3-27b-it-qat-q4_0-q3_k_m │ REASON-column-words.txt       │ 𐄂         │ 14.53s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gemma-3-27b-it-qat-q4_0-q3_k_m │ REASON-ramarronero.txt        │ 𐄂         │ 5.85s           │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ 1-capital-italy.txt           │ 🮱         │ 26.83s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ BIGCONTEXT-kuleba.txt         │ 🮱         │ 48.03s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ CODING-typescript-rust.txt    │ 🮱         │ 33.07s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ EXTRACT-USDT-APY.txt          │ 🮱         │ 133.22s         │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ KNOWLEDGE-translate-pesca.txt │ 🮱         │ 18.67s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ MATH-battery-discarge.txt     │ 🮱         │ 29.25s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ REASON-column-words.txt       │ 🮱         │ 81.82s          │
├────────────────────────────────┼───────────────────────────────┼───────────┼─────────────────┤
│ gpt-oss-20b-mxfp4              │ REASON-ramarronero.txt        │ 🮱         │ 16.90s          │
╘════════════════════════════════╧═══════════════════════════════╧═══════════╧═════════════════╛

Model Performance Summary
╒════════════════════════════════╤═══════════════════════════╤═════════════════════╕
│ Model                          │ Correct                   │ Avg Response Time   │
╞════════════════════════════════╪═══════════════════════════╪═════════════════════╡
│ Qwen3-4B-IQ4_NL                │ 5/8 (62.5%) [██████░░░░]  │ 87.92s              │
├────────────────────────────────┼───────────────────────────┼─────────────────────┤
│ gemma-3-27b-it-qat-q4_0-q3_k_m │ 6/8 (75.0%) [███████░░░]  │ 112.57s             │
├────────────────────────────────┼───────────────────────────┼─────────────────────┤
│ gpt-oss-20b-mxfp4              │ 8/8 (100.0%) [██████████] │ 48.47s              │
╘════════════════════════════════╧═══════════════════════════╧═════════════════════╛
```

## Troubleshooting

- **API Errors**: Ensure your endpoint is running and accessible. Check the URL and model names in `.env`.
- **Evaluator Failures**: If using `MODEL_EVALUATOR`, it should return "CORRECT" or "INCORRECT". The script now handles variations like "not correct".
- **No Matching Answers**: The script skips prompts without corresponding answer files.
- **Dependencies**: If uv is not installed, download it from [astral-sh/uv](https://github.com/astral-sh/uv).
- **Customization**: Modify `main.py` for advanced features, like adding more metrics or output formats.

## Testing

The project includes comprehensive testing with 80%+ code coverage.

### Backend Tests (pytest)

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_main.py

# Run with coverage report
uv run pytest --cov=. --cov-report=html
```

### Frontend Tests (Vitest)

```bash
cd frontend

# Run all tests
npm run test:run

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test
```

Test structure: `tests/unit/` for backend, `src/test/` for frontend. Both support unit and integration tests with proper mocking.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`uv run pytest && cd frontend && npm run test:run`)
5. Follow existing code style and patterns
6. Submit a pull request

For questions or issues, please open a GitHub issue.

## License

MIT License. See [`LICENSE`](LICENSE) for details.
