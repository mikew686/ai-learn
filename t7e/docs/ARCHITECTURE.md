# t7e app at ai-learn/t7e

## Subdirectory and CI/CD

**Yes — the app can live in a subdirectory and still be used for CI/CD.** Common approaches:

- **Monorepo workflow**: CI runs from repo root; workflows are scoped to `t7e/` via paths (e.g. `paths: ['t7e/**']` in GitHub Actions) so only changes under `t7e` trigger builds.
- **Docker**: Build from **project root** (ai-learn): `docker build .` (Dockerfile is at repo root). Gunicorn runs with CWD = project root.
- **k8s/Kustomize**: Manifests live under `t7e/k8s/` (base + overlays); CI builds the image, then applies with `kubectl apply -k t7e/k8s/overlays/<env>`.

No structural change is needed later; CI/CD simply uses the subdir as the app root.

---

## Proposed layout (all under ai-learn repo root)

```
ai-learn/
  t7e/
    src/
      app/           # Flask app (factory, routes, config)
      rqworker/      # RQ (Redis Queue) worker entrypoint
      utils/         # Shared helpers
    migrations/      # Alembic
    k8s/             # Kustomize: base + overlays (e.g. dev, prod)
      base/
      overlays/
        dev/
        prod/
    docs/            # Project docs (e.g. ARCHITECTURE.md)
    docker-compose.yml   # optional local dev
    requirements.txt    # synced from pyproject.toml; used by Docker builds
    alembic.ini
    .env.example
    README.md
```

- **Dependencies**: Single `pyproject.toml` at **repo root** (ai-learn); it lists t7e deps and exposes packages `app`, `rqworker`, `utils` from `t7e/src`. Local: `pip install -e .` from repo root. Docker: `pip install -e .` in image.
- **Database**: **PostgreSQL** with the **pgvector** extension. The service uses Postgres as its primary database; pgvector supports vector similarity (e.g. for embeddings). Alembic migrations run against this database.
- **Flask**: App factory in `src/app/__init__.py`, blueprints/routes under `src/app/`, config via env (e.g. `src/app/config.py`).
- **RQ worker**: A second app in `src/rqworker/` runs as an **RQ (Redis Queue)** worker, consuming jobs from Redis. The Flask app enqueues jobs; the worker runs them (e.g. heavy or async tasks). **Redis** is required for the queue; same codebase, different process/container (e.g. `rq worker` or `python -m src.rqworker`).
- **Alembic**: `alembic.ini` and `migrations/` at `t7e/`; `env.py` uses a single metadata from your app's models (e.g. `from src.app.models import Base`). Run from `t7e/`: `alembic upgrade head`.
- **Docker**: **Dockerfile at repo root.** Build from project root: `docker build .`. Image has the repo; gunicorn runs with **CWD = project root** (paths and config resolve correctly). (1) **Web**: WORKDIR = project root, `pip install -e .`, runs **gunicorn** (`app:create_app`). (2) **Worker**: same image, CMD runs the RQ worker (`python -m rqworker`). Local dev: `pip install -e .` then `python -m app`; `docker-compose.yml` can bring up Postgres (pgvector), Redis, web, and worker.
- **k8s (Kustomize)**: Base manifests under `t7e/k8s/base/`: Deployment(s) for web (gunicorn) and rqworker, Service(s), optional Ingress for web. Overlays patch image tag, replicas, env. Postgres and Redis may be in-cluster (e.g. StatefulSet/Deployment + Service) or external; Postgres must have pgvector enabled. Apply with `kubectl apply -k t7e/k8s/overlays/<env>`.
- **CI/CD**: One workflow (e.g. `.github/workflows/t7e.yml`) that:
  - Triggers on changes under `t7e/**`
  - Sets working dir to repo root for build
  - Builds image with `docker build .` (Dockerfile at root), pushes to registry
  - Runs tests from `t7e/` if you add them
  - Deploys via `kubectl apply -k t7e/k8s/overlays/<env>` (or `kustomize build t7e/k8s/overlays/<env> | kubectl apply -f -`)

---

## Implementation order

1. **Scaffold** – Create `t7e/` with `src/app`, `src/rqworker`, and `src/utils` (packages with `__init__.py`), minimal Flask factory and one health route, `.env.example`. Dependencies and package layout live in repo root `pyproject.toml`. Include RQ and Redis connection config.
2. **Database & Alembic** – Postgres with pgvector (local via docker-compose or external). Add `alembic.ini` and `migrations/` in `t7e/`, wire `env.py` to app config and (when you add them) app models; ensure pgvector is available for any vector columns.
3. **Docker** – Dockerfile at **repo root**; build with `docker build .`, WORKDIR = project root. Web: gunicorn; worker: same image, `python -m rqworker`. Optional `docker-compose.yml` for local Postgres (pgvector), Redis, web, and worker.
4. **k8s (Kustomize)** – Add `t7e/k8s/base/` with Deployment for web, Deployment for rqworker, Service(s), optional Ingress; base for Postgres/Redis or assume external. Overlays (e.g. `overlays/dev`, `overlays/prod`) patch image and env. Apply with `kubectl apply -k t7e/k8s/overlays/<env>`.
5. **CI/CD** – Add workflow under `.github/workflows/` that triggers on `t7e/**`, builds from **project root** (`docker build .`), and deploys using `kubectl apply -k t7e/k8s/overlays/<env>`.

This keeps the app self-contained under `ai-learn/t7e` and fully compatible with CI/CD.
