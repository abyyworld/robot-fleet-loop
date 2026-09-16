.DEFAULT_GOAL := help
# Prefer 3.12, but fall back to whatever python3 is on PATH. A Mac with
# Homebrew Python but no python3.12 should not fail on the first command
# in the README. Override with `make PY=python3.11 install`.
PY ?= $(shell command -v python3.12 2>/dev/null || command -v python3 2>/dev/null || echo python3)
VENV := .venv
BIN := $(VENV)/bin
ROOT ?= var/fleet
ROUNDS ?= 8

# The device half of the loop. A local checkout is preferred when there is one,
# so the two repos can be developed together; otherwise pip resolves the git URL
# in pyproject.toml. Both common layouts are checked — side by side, or one
# directory up each — because which one you have depends on how you cloned them
# and neither is worth a manual step.
EDGE ?= $(firstword $(wildcard ../edge-policy-runtime ../../edge-policy-runtime \
                               ../../p4/edge-policy-runtime))

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install, using a sibling edge-policy-runtime checkout if present
	@if [ -n "$(EDGE)" ] && [ -d "$(EDGE)" ]; then \
		echo "using local checkout at $(EDGE)"; \
		$(BIN)/pip install -e "$(EDGE)"; \
		$(BIN)/pip install -e . --no-deps; \
		$(BIN)/pip install 'numpy>=1.24' 'pydantic>=2.5' 'typer>=0.9' 'rich>=13.0'; \
		$(BIN)/pip install 'pytest>=7.4' 'pytest-cov>=4.1' 'ruff>=0.3'; \
	else \
		$(BIN)/pip install -e '.[dev]'; \
	fi

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Lint and format-check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

.PHONY: fmt
fmt: ## Auto-format
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: anchors
anchors: ## Confirm the task discriminates before believing any number from it
	$(BIN)/fleet-loop anchors

.PHONY: loop
loop: ## The whole loop: three nodes, selective sync, retrain, canary, promote
	$(BIN)/fleet-loop run --rounds $(ROUNDS) --root $(ROOT)

.PHONY: rollback
rollback: ## Ship a release every pre-flight check approves, and watch it come back
	$(BIN)/fleet-loop regression --root var/regression

.PHONY: power
power: ## How many episodes a fleet comparison actually needs
	$(BIN)/fleet-loop power

.PHONY: clean
clean: ## Remove fleet state and reports
	rm -rf var reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache
