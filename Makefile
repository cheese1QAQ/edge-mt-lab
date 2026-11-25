UV ?= uv

.PHONY: help sync lint format typecheck test check doctor bench-dry

help:
	@echo "Edge-MT-Lab Research Commands:"
	@echo "  make sync       Install Python dependencies & dev tools via uv"
	@echo "  make lint       Run ruff lint & format check (src + tests)"
	@echo "  make format     Auto-format Python code with ruff"
	@echo "  make typecheck  Run strict mypy type checking"
	@echo "  make test       Run pytest suite with coverage"
	@echo "  make check      Run full CI gate (lint + typecheck + test)"
	@echo "  make doctor     Inspect hardware and compute 2-4GB VRAM budgets"
	@echo "  make bench-dry  Validate eval dataset & dry-run benchmark harness"

sync:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov=edge_mt_lab --cov-report=term-missing

check: lint typecheck test

doctor:
	$(UV) run edge-mt-lab doctor

bench-dry:
	$(UV) run edge-mt-lab bench --dry-run
