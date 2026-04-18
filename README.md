# Movie Recommendation System

This repository contains a movie recommendation service built around collaborative filtering, model retraining, data-quality checks, offline and online evaluation, provenance logging, and a FastAPI inference layer.

The project is structured more like an applied ML system than a single notebook. It includes model artifacts, serving code, CI automation, evaluation scripts, and operational checks for monitoring recommendation quality over time.

## Overview

The system serves movie recommendations through a FastAPI API and supports multiple model variants for experimentation and online evaluation.

Core pieces in the repository:

- a FastAPI recommendation service in `app/`
- collaborative filtering models based on Surprise `SVD`
- trained and retrained model artifacts
- offline and online evaluation utilities
- data-quality and drift checks
- provenance logging for recommendation requests
- Docker and Jenkins-based deployment workflow

## Main Features

- Recommend top movies for a user through `/recommend/{user_id}`
- Load and serve multiple model versions for A/B-style routing
- Track provenance for each response, including model version, data version, and pipeline commit
- Evaluate recommendation quality offline with saved reports
- Track online metrics and telemetry over time
- Run data schema validation and drift checks
- Package the service with Docker and deploy it through Jenkins

## Tech Stack

- Python
- FastAPI
- Surprise
- pandas
- scikit-learn
- Docker
- Jenkins
- pytest
- LaunchDarkly

## Repository Structure

```text
app/
  server.py                 FastAPI app and model selection logic
  models.py                 recommendation model loading, training, and inference
  schema.py                 API response schema
  provenance.py             provenance logging

data/
  final_processed_data.csv  processed ratings dataset
  final_data.csv            dataset variant used by analysis/evaluation

evaluation/
  offline/                  offline evaluation scripts and saved results
  online/                   telemetry and online evaluation utilities

data_quality/
  check_data_quality.py     schema/data checks
  check_drift.py            drift detection workflow

trained_models/
retrained_models/
  model artifacts used by the API and retraining experiments

tests/
  API, model, and data pipeline tests
```

## API

### Get Recommendations

```http
GET /recommend/{user_id}
```

Example:

```bash
curl http://localhost:8082/recommend/186719
```

### API Docs

When the server is running:

- `http://localhost:8082/docs`
- `http://localhost:8082/redoc`

## How It Works

### Model Serving

The API loads one baseline model and multiple retrained model snapshots at startup. LaunchDarkly is used to choose which model variant handles a request, which makes it possible to compare recommendation behavior across versions.

### Recommendation Flow

1. Receive a user ID through the API.
2. Select a model variant.
3. Generate recommendations from unseen movies for that user.
4. Log provenance metadata for the response.
5. Return the movie list through the API schema.

### Provenance

Each request logs:

- timestamp
- user ID
- model version
- data version
- pipeline commit
- generated recommendations

That makes the service easier to audit and debug when recommendation quality changes.

## Running Locally

### Option 1: Docker

```bash
docker compose build
docker compose up
```

The API will be available on `http://localhost:8082`.

### Option 2: Local Python Environment

Use Python `3.10+` to match the Docker runtime and dependency set.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 8082 --reload
```

If you want LaunchDarkly-based model routing enabled, set:

```bash
export LD_SDK_KEY=your_key_here
```

## Testing

Run the test suite with:

```bash
pytest
```

## CI/CD

The repository includes a Jenkins pipeline that:

- creates a virtual environment
- installs dependencies
- runs formatting checks with `black`
- runs tests with `pytest`
- records commit metadata for deployment
- builds and deploys the Docker container with `docker compose`

## Evaluation and Quality Checks

This repo includes work beyond plain inference:

- offline evaluation in `evaluation/offline/`
- online metric tracking in `evaluation/online/`
- data schema and drift checks in `data_quality/`
- fairness-related analysis in `analysis/fairness/`

That makes it a stronger ML systems project than a simple recommendation demo.

## Notes

- The repository contains model artifacts and generated evaluation outputs, so it is heavier than a source-only service repository.
- Some online evaluation and feature-routing behavior depends on external configuration such as LaunchDarkly keys.
- The strongest part of this repo is the full ML workflow around serving, monitoring, and retraining, not just the API endpoint.
