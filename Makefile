.PHONY: help install install-dev install-jupyter install-all clean lint format type-check test test-cov notebooks deploy-server run-server check pre-commit update-deps lock-deps info

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python and UV paths
PYTHON := python3
UV := uv

help: ## Display this help message
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║        Generative AI Course - Production Build System           ║$(NC)"
	@echo "$(BLUE)║                   Author: Ruslan Magana                          ║$(NC)"
	@echo "$(BLUE)║                Website: ruslanmv.com                             ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install core dependencies using uv
	@echo "$(BLUE)Installing core dependencies with uv...$(NC)"
	$(UV) pip install -e .
	@echo "$(GREEN)✓ Core installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(UV) pip install -e ".[dev]"
	@echo "$(GREEN)✓ Development dependencies installed!$(NC)"

install-jupyter: ## Install Jupyter and notebook dependencies
	@echo "$(BLUE)Installing Jupyter dependencies...$(NC)"
	$(UV) pip install -e ".[jupyter]"
	@echo "$(GREEN)✓ Jupyter dependencies installed!$(NC)"

install-all: ## Install all dependencies (core + dev + jupyter)
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	$(UV) pip install -e ".[all]"
	@echo "$(GREEN)✓ All dependencies installed!$(NC)"

lock-deps: ## Generate uv.lock file with pinned dependencies
	@echo "$(BLUE)Locking dependencies...$(NC)"
	$(UV) pip compile pyproject.toml -o requirements.lock
	@echo "$(GREEN)✓ Dependencies locked!$(NC)"

update-deps: ## Update all dependencies to latest compatible versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(UV) pip install --upgrade -e ".[all]"
	@echo "$(GREEN)✓ Dependencies updated!$(NC)"

clean: ## Clean build artifacts, cache files, and temporary files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code with black...$(NC)"
	black src/ Deployment/ trainer/ LLMs/ Finetuning/ Multimodal/ Transformers/ --exclude="checkpoints|outputs|backups"
	@echo "$(BLUE)Auto-fixing with ruff...$(NC)"
	ruff check --fix src/ Deployment/ trainer/ LLMs/ Finetuning/ Multimodal/ Transformers/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

lint: ## Run linting with ruff
	@echo "$(BLUE)Running ruff linter...$(NC)"
	ruff check src/ Deployment/ trainer/ LLMs/ Finetuning/ Multimodal/ Transformers/
	@echo "$(GREEN)✓ Linting complete!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	mypy src/ Deployment/ trainer/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

pre-commit: format lint type-check ## Run all pre-commit checks (format, lint, type-check)
	@echo "$(GREEN)✓ All pre-commit checks passed!$(NC)"

check: lint type-check test ## Run all quality checks (lint, type-check, test)
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

notebooks: ## Launch Jupyter Lab for interactive development
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

deploy-server: ## Deploy the FastAPI inference server
	@echo "$(BLUE)Deploying FastAPI inference server...$(NC)"
	cd Deployment && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

run-server: ## Run the inference server in production mode
	@echo "$(BLUE)Running production server...$(NC)"
	cd Deployment && uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

info: ## Display environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "  Python version: $$($(PYTHON) --version)"
	@echo "  UV version: $$($(UV) --version)"
	@echo "  Working directory: $$(pwd)"
	@echo "  Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo ""
	@echo "$(BLUE)Installed packages:$(NC)"
	@$(UV) pip list | head -20
	@echo ""

# Training targets
train-sft: ## Run supervised fine-tuning
	@echo "$(BLUE)Running supervised fine-tuning...$(NC)"
	cd trainer && bash sft.sh

train-pretrain: ## Run pre-training
	@echo "$(BLUE)Running pre-training...$(NC)"
	cd trainer && bash pretrain.sh

download-model: ## Download required models
	@echo "$(BLUE)Downloading models...$(NC)"
	cd trainer && bash download_model.sh

download-dataset: ## Download required datasets
	@echo "$(BLUE)Downloading datasets...$(NC)"
	cd trainer && bash download.sh

# Documentation
docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	mkdocs serve -a 0.0.0.0:8080

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

# Docker targets (future enhancement)
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t generative-ai-course:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm --gpus all -p 8000:8000 generative-ai-course:latest

# Utility targets
setup-git-hooks: ## Setup git hooks for pre-commit
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Git hooks installed!$(NC)"

version: ## Display version information
	@echo "$(BLUE)Generative AI Course v1.0.0$(NC)"
	@echo "Author: Ruslan Magana"
	@echo "Website: ruslanmv.com"
	@echo "License: Apache-2.0"
