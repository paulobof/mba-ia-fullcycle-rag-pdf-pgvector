.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help setup ingest chat test test-integration lint format typecheck check db-up db-down requirements

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-14s %s\n", $$1, $$2}'

setup: ## Instala as dependências (runtime + dev) com uv
	$(UV) sync --all-groups

db-up: ## Sobe o Postgres com pgVector
	docker compose up -d

db-down: ## Derruba o banco (use ARGS=-v para apagar o volume)
	docker compose down $(ARGS)

ingest: ## Ingere o PDF configurado em PDF_PATH
	$(UV) run python src/ingest.py

chat: ## Abre o chat no terminal
	$(UV) run python src/chat.py

test: ## Roda a suíte de testes com cobertura
	$(UV) run pytest

test-integration: ## Roda os testes de integração (exige make db-up)
	$(UV) run pytest -m integration --no-cov

lint: ## Verifica lint (ruff)
	$(UV) run ruff check .

format: ## Formata o código (ruff)
	$(UV) run ruff format .

typecheck: ## Verifica tipos (mypy strict)
	$(UV) run mypy

check: lint typecheck test ## Roda lint, tipos e testes

requirements: ## Regenera o requirements.txt a partir do uv.lock
	$(UV) export --no-hashes --no-dev --no-emit-project -o requirements.txt
