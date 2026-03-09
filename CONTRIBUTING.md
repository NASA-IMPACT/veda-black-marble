# Contributing to VEDA Black Marble

Thank you for your interest in contributing! This guide will help you get started with development.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git

### Development Setup

Install your dependencies 

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

## Development Workflow

### Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_file.py

# Run specific test
uv run pytest tests/test_file.py::test_function
```

### Code linting and formatting

Ruff is used for code quality and formatting:

```bash
uv run ruff check --fix && uv run ruff format
```

### Pre-commit

Pre-commit hooks can be installed:

```bash
# Install
uv run pre-commit install
# Run
uv run pre-commit run --all-files
```

### Type Checking

Basedpyright is used for type checking:

```bash
# Check the entire codebase
uv run basedpyright blackmarble/ tests/
```

## Code Guidelines

### Dependencies

- Update `pyproject.toml` with version constraints
- Keep dependencies up-to-date and secure

## Documentation

- `/docs/RELEASE_NOTES.md` documents changes made to the pipeline
- Update relevant documentation in `/docs` for significant changes
- Update `README.md` for user-facing changes
- Include docstrings in code using standard Google or NumPy style

## Questions?

- Check existing [GitHub Issues](https://github.com/NASA-IMPACT/veda-black-marble/issues)
- Review the [README](README.md) and docs in `/docs`
- Open a new issue for questions or discussions
