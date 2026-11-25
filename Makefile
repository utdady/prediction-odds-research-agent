SHELL := bash

ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
else
  VENV_BIN := .venv/bin
endif

VENV_PYTHON := $(VENV_BIN)/python
ACTIVATE := . $(VENV_BIN)/activate

setup:
	python -m venv .venv && $(ACTIVATE) && python -m pip install -U pip && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

db_setup:
	$(ACTIVATE) && python setup_db_interactive.py

migrate:
	$(ACTIVATE) && alembic upgrade head

run_all:
	export PYTHONPATH=src && $(ACTIVATE) && python -m pipelines.run_all

api:
	$(ACTIVATE) && uvicorn src.app.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	$(ACTIVATE) && streamlit run src.app.dashboard.Home --server.port 8501

test:
	$(ACTIVATE) && pytest -q