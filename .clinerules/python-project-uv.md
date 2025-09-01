## Brief overview
  This set of guidelines is specific to Python projects worked on within this context. It outlines the preference for using `uv` as the package manager and installer over `pip`.

## Package Management
  - Use `uv` for all Python package management tasks instead of `pip`.
  - This includes installing dependencies (`uv add <package>`), uninstalling packages (`uv remove <package>`), and creating/updating lock files (`uv lock`).
  - When setting up a new Python project, initialize it with `uv init` to create a `pyproject.toml` file configured for `uv`.
  - Prefer using `uv run` to execute Python scripts or commands within the project's virtual environment to ensure dependencies are available.

## Dependency Management
  - Define project dependencies in the `pyproject.toml` file under the `[project.dependencies]` or `[tool.uv.dev-dependencies]` sections.
  - Rely on `uv.lock` to ensure reproducible builds across different environments.
  - When adding new dependencies, use `uv add <package_name>` or `uv add --group dev <package_name>` for development dependencies.
  - Remove unused dependencies with `uv remove <package_name>`.

## Virtual Environment
  - `uv` automatically manages a virtual environment. There's no need to manually create or activate one with `python -m venv` or `source venv/bin/activate`.
  - Use `uv sync` to install dependencies from the `pyproject.toml` and `uv.lock` files, creating or updating the virtual environment as needed.
