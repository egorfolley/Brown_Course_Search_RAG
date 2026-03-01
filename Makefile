.PHONY: help install run-local stop-local run-docker run-docker-fast prep-docker build test clean logs stop

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies locally"
	@echo "  make run-local        - Run backend + frontend locally (FAST, one command)"
	@echo "  make stop-local       - Stop local backend + frontend"
	@echo "  make run-docker-fast  - Run Docker with pre-generated data (RECOMMENDED)"
	@echo "  make run-docker       - Run entire stack with Docker Compose (slow first run)"
	@echo "  make prep-docker      - Generate data locally for fast Docker startup"
	@echo "  make build            - Build Docker images"
	@echo "  make test             - Run test suite"
	@echo "  make clean            - Remove data files (force fresh scrape)"
	@echo "  make logs             - View Docker logs"
	@echo "  make stop             - Stop Docker containers"

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "⚠️  Playwright install skipped (not needed for Docker). Run manually if needed:"
	@echo "    . .venv/bin/activate && playwright install chromium"

run-local:
	./start-local.sh

stop-local:
	@echo "🛑 Stopping local services..."
	@pkill -f "python app/app.py" 2>/dev/null || true
	@pkill -f "streamlit run" 2>/dev/null || true
	@pkill -f "uvicorn" 2>/dev/null || true
	@echo "✅ Services stopped"

prep-docker:
	@echo "📦 Generating data locally (5-10 minutes)..."
	@echo "Press Ctrl+C when you see 'Uvicorn running on http://0.0.0.0:8000'"
	python app/app.py

run-docker-fast: prep-docker
	@echo "🐳 Starting Docker with pre-generated data (~30 seconds)..."
	docker-compose up --build

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
