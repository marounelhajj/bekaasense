# ======================================================================
# BekaaSense — reproducibility Makefile
# Rubric tie-in: EN3 reproducible environment + run path.
# ======================================================================

.PHONY: help install data train evaluate test serve up down logs \
        migrate collectstatic clean lint shell

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?##' '{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------- Local (non-Docker) ----------

install:  ## Install Python dependencies into the active environment
	pip install -r requirements.txt

data:  ## Generate the synthetic processed CSV (for demo / CI)
	python scripts/generate_synthetic.py

ingest:  ## Build canonical CSV from raw station files in data/raw/
	python scripts/build_canonical.py

train:  ## Train every model (baselines + ML) and write metrics
	python -m model_engine.train

evaluate:  ## Print the leaderboard from the last training run
	python scripts/show_leaderboard.py

test:  ## Run the test suite
	pytest -v --cov=data_ingestion --cov=model_engine --cov=api

migrate:  ## Apply Django migrations
	python manage.py migrate --noinput

collectstatic:  ## Collect static assets
	python manage.py collectstatic --noinput

serve:  ## Serve locally on :8000 (dev mode)
	python manage.py runserver 0.0.0.0:8000

shell:  ## Django shell
	python manage.py shell

lint:  ## Check Python syntax (fast)
	python -m compileall -q data_ingestion model_engine api dashboard bekaasense

# ---------- Docker ----------

up:  ## Bring the full stack up (build + start)
	docker compose up --build -d
	@echo "Open http://localhost:8000"

down:  ## Tear the stack down
	docker compose down

logs:  ## Tail service logs
	docker compose logs -f web

docker-train:  ## Run training inside the web container
	docker compose exec web python -m model_engine.train

docker-test:  ## Run tests inside the web container
	docker compose exec web pytest -v

# ---------- Housekeeping ----------

clean:  ## Remove generated artefacts (keeps raw data)
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov staticfiles
	rm -f  db.sqlite3
	rm -rf model_engine/artifacts/*.joblib
	rm -f  results/metrics/*.csv results/metrics/*.json
