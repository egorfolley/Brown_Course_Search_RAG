.PHONY: help install run-local run-docker build test clean logs stop

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies locally"
	@echo "  make run-local    - Run backend and frontend locally (requires 2 terminals)"
	@echo "  make run-docker   - Run entire stack with Docker Compose"
	@echo "  make build        - Build Docker images"
	@echo "  make test         - Run test suite"
	@echo "  make clean        - Remove data files (force fresh scrape)"
	@echo "  make logs         - View Docker logs"
	@echo "  make stop         - Stop Docker containers"

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && playwright install chromium

run-local:
	@echo "Start backend: python app/app.py"
	@echo "Start frontend (new terminal): streamlit run frontend/ui.py"

run-docker:
	docker-compose up --build

build:
	docker-compose build

test:
	pytest -v

logs:
	docker-compose logs -f

stop:
	docker-compose down

clean:
	rm -rf data/*.json data/*.npy data/faiss.index data/metadata.json
	@echo "Data files removed. Next run will rebuild from scratch."
