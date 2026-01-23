SHELL := bash

setup:
	python -m venv .venv && . .venv/Scripts/activate && python -m pip install -U pip && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

db_setup:
	. .venv/Scripts/activate && python setup_db_interactive.py

migrate:
	. .venv/Scripts/activate && alembic upgrade head

run_all:
	. .venv/Scripts/activate && python -m pipelines.run_all

api:
	. .venv/Scripts/activate && uvicorn src.app.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	. .venv/Scripts/activate && streamlit run src.app.dashboard.Home --server.port 8501

test:
	. .venv/Scripts/activate && pytest -q

