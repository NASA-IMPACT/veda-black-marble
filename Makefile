.PHONY: lint typecheck audit baseline format install-hooks

lint:        ## Run ruff linter (non-zero exit)
	uv run ruff check blackmarble

typecheck:   ## Run pyright strict type checking
	uv run pyright blackmarble

audit:       ## Run all audit scripts to generate metrics
	uv run python scripts/generate_baseline_metrics.py

baseline: audit  ## Alias for audit

format:      ## Format code with black and isort
	uv run black blackmarble
	uv run isort blackmarble

install-hooks:  ## Install pre-commit hooks
	uv run pre-commit install

help:        ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'