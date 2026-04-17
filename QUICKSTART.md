# BekaaSense — Deployment Cheat Sheet

Step-by-step commands to go from the ZIP file to a running, committed, and
published project. Copy-paste friendly.

---

## 0. Prerequisites

You need installed on your machine:

- **Python 3.11** (check: `python --version`)
- **Git** (check: `git --version`)
- **Docker Desktop** (check: `docker --version`)
- A **GitHub** account

---

## 1. Unpack and test locally (no Docker)

```bash
# Unzip wherever you want
unzip bekaasense.zip
cd bekaasense

# Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies (takes 2-4 minutes)
pip install -r requirements.txt

# Generate the demo dataset
python scripts/generate_synthetic.py

# Train every model (RF, XGBoost, 3 baselines, classifier)
python -m model_engine.train

# Run the tests
pytest -v

# Apply Django migrations
python manage.py migrate

# Start the dev server
python manage.py runserver 0.0.0.0:8000
```

Then open **http://localhost:8000** in your browser. You should see the
BekaaSense dashboard with the historical trend, forecast, SHAP chart, and
leaderboard.

Press **Ctrl+C** to stop the server.

---

## 2. Use your real data (instead of synthetic)

```bash
# Put your 4 raw station files here:
#   data/raw/Ammik.xlsx
#   data/raw/Doures.xlsx
#   data/raw/Ras_Baalbeck.xlsx
#   data/raw/Tal_Amara.xlsx

python scripts/build_canonical.py
python -m model_engine.train
python manage.py runserver 0.0.0.0:8000
```

The loader is tolerant of common column-name variations. If a file
rejects, check the log — it will tell you which canonical column was
missing and which aliases were tried.

---

## 3. Create the GitHub repository

```bash
# From inside the bekaasense/ folder:
git init
git add .
git commit -m "Initial commit — BekaaSense MVP"
git branch -M main
```

On GitHub, create a new **empty** repo called `bekaasense` (no README, no
.gitignore, no license — those are already in the project).

Then link and push:

```bash
git remote add origin https://github.com/<your-username>/bekaasense.git
git push -u origin main
```

GitHub Actions (CI) will automatically run the test suite and training
pipeline on every push — see `.github/workflows/ci.yml`.

---

## 4. Run with Docker

```bash
cp .env.example .env
# (Open .env and change DJANGO_SECRET_KEY to any long random string.)

# Build and start the full stack
docker compose up --build
```

First build: ~3 minutes. Then open **http://localhost:8000**.

To train inside the container:

```bash
docker compose exec web python scripts/generate_synthetic.py
docker compose exec web python -m model_engine.train
```

Stop with **Ctrl+C**, or in another terminal:

```bash
docker compose down
```

---

## 5. All-in-one with Make

```bash
make install      # pip install
make data         # generate synthetic data
make train        # train all models
make test         # pytest
make serve        # dev server :8000
make up           # docker compose up --build
make down         # docker compose down
make evaluate     # print leaderboard
make help         # list all targets
```

---

## 6. Quick API smoke test

With the server running:

```bash
# Health
curl http://localhost:8000/health/

# List stations (with latest aridity + viability)
curl http://localhost:8000/api/stations/ | python -m json.tool

# 12-month forecast for Ammik with 90% interval
curl -X POST http://localhost:8000/api/predict/ \
     -H "Content-Type: application/json" \
     -d '{"station":"Ammik","horizon_months":12,"alpha":0.1}' \
     | python -m json.tool

# Model leaderboard
curl http://localhost:8000/api/leaderboard/ | python -m json.tool

# SHAP explanation for the latest Ammik prediction
curl -X POST http://localhost:8000/api/explain/ \
     -H "Content-Type: application/json" \
     -d '{"station":"Ammik","top_k":8}' \
     | python -m json.tool
```

---

## 7. Cloud deployment (one option: Render.com)

1. Push your repo to GitHub (step 3).
2. Sign up at **render.com** with your GitHub account.
3. Click **New → Web Service**, pick your repo.
4. Settings:
   - **Environment**: Docker
   - **Region**: closest to you
   - **Instance type**: Free (sufficient for the demo)
   - **Environment variables**: paste `DJANGO_SECRET_KEY` from your `.env`
5. Click **Deploy**. First build: ~5 minutes.

Your URL will be `https://bekaasense-<random>.onrender.com` — submit this
link in your project write-up for the **EN5 running deployed artefact**
rubric criterion.

Alternative free platforms: **Railway.app**, **Fly.io**, **Hugging Face
Spaces** (the last needs a small `Spacefile` instead of `render.yaml`).

---

## 8. Final submission checklist

Before your poster / final presentation, verify:

- [ ] `README.md` — quickstart works on a clean machine
- [ ] `readme_correction.md` — every rubric code has a file mapping
- [ ] `docs/ARCHITECTURE.md`, `DATA.md`, `EVALUATION.md`, `LIMITATIONS.md` exist
- [ ] `results/metrics/leaderboard.csv` contains all 6 models
- [ ] `pytest` passes 17/17 (the 18th skips when `data/processed/bekaa_valley_clean.csv` exists)
- [ ] `docker compose up` starts the stack cleanly
- [ ] `/health/` returns 200
- [ ] Dashboard at `/` renders the chart, SHAP panel, and viability light
- [ ] GitHub Actions CI is green
- [ ] (Optional bonus) Cloud deployment URL works

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
→ You ran an old Python or skipped `pip install -r requirements.txt`.

**`sqlite3.OperationalError: no such table`**
→ You forgot `python manage.py migrate`.

**Dashboard shows "No leaderboard"**
→ You forgot `python -m model_engine.train`.

**Docker build fails on `xgboost`**
→ xgboost needs `libgomp1` at runtime. The Dockerfile already installs
it. Make sure you didn't edit that line.

**Port 8000 in use**
→ Stop whatever is using it: `docker ps` → `docker stop <id>`, or pick a
different host port: `ports: ["8001:8000"]` in `docker-compose.yml`.

**Port 8000 in use on Windows**
→ In PowerShell: `netstat -ano | findstr :8000` → note the PID →
`taskkill /F /PID <pid>`.
