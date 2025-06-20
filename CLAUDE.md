# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Git Commit Guidelines

1. Keep commit messages short and descriptive
2. NEVER include attribution like "Generated with Claude Code" or Co-Authored-By
3. Focus on WHAT changed, not HOW it was created
4. Use trailers only for:
   - `Reported-by:<name>` for user-reported issues
   - `Github-Issue:#<number>` for issue references

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add <package>`
   - Installation (dev): `uv add --dev <package>`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --upgrade package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions should be focused and small
   - Follow existing patterns exactly

3. Testing Requirements
   - Framework: `uv run pytest`
   - Static analysis: `uv run pre-commit run --all-files`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests
   - NEVER cheat on tests. Fix the code, not the tests.

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting


## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Descriptive Names**: Use clear variable/function names
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organisation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## Core Components

- `config.py`: Configuration management
- `daemon.py`: Main daemon
[etc... fill in here]

## Code Formatting

1. Ruff
   - Format: `uv run ruff format .`

2. Type Checking
   - Tool: `uv run pyright`
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

3. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Tools: Ruff Python, MyPy

## Tool Use (MCP)

1. Basic-Memory
   - Take notes whenever dealing with complicated bugs. Use these notes to keep track of progress
   - Plan out changes with notes using checkboxes as TODO items

2. Docfork
   - Use docfork to look up documentation
