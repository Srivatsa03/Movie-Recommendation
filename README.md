# Movie Recommendation System (MLOps)

**A recommendation engine shipped the way a real ML system should be: trained, served, monitored, tested, and guarded, not a notebook that works once.**

## What it is

The full ML lifecycle around a movie recommender. The model itself is the easy part; this project is about everything that surrounds it in production, from CI/CD to drift detection to abuse protection. Built across a 6-member team.

## The lifecycle

```
train ──> serve (FastAPI + Docker) ──> Jenkins CI/CD ──> deploy
   │                                        │
   ├─ A/B testing                           ├─ SonarQube quality gate (100% of PRs)
   ├─ drift detection                       │
   ├─ fairness checks                       v
   └─ rate limiting + poisoning detection   Prometheus + Grafana (5+ metrics, alerts)
```

- **Serving:** FastAPI behind Docker, deployed through a full Jenkins pipeline.
- **Trust & safety from day one:** A/B testing, drift detection, fairness checks, rate limiting, and data-poisoning detection, all built in rather than retrofitted.
- **Observability:** Prometheus + Grafana tracking 5+ performance metrics with automated alerting and deployment validation.
- **Quality:** SonarQube gates enforced on 100% of PRs, with full audit traceability across environments.

## Results

- **RMSE 0.58.**
- **65% faster inference.**

## Stack

`FastAPI` · `Docker` · `Jenkins` · `Prometheus` · `Grafana` · `SonarQube` · `Python` · `scikit-learn`
