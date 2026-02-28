# mw2

Flask app for mw2; lives under `ai-learn/mw2`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for layout and deployment.

## Local run

Install from repo root so `app`, `utils`, and `rqworker` are on the path:

```bash
# From ai-learn repo root:
pip install -e .

# Run the app (from any directory):
python -m app
```

Then open http://127.0.0.1:5000/. The index page shows a hello-world message and the caller IP. You can also `import utils` and `from app import create_app`; packages come from `mw2/src` (defined in root `pyproject.toml`).

## Kustomize (local overlay)

Redis and PostgreSQL (with pgvector) run in the `mw2` namespace with persistent volumes. See **[docs/k8s_local.md](docs/k8s_local.md)** for how to start, stop, and expose them on standard ports (6379, 5432).

```bash
# Build the local pgvector Postgres image first (required for the local overlay):
mw2/k8s/scripts/build-postgres-image.sh

kubectl apply -k mw2/k8s/overlays/local
# Then port-forward to use Redis and Postgres on this machine (see docs/k8s_local.md).
```

## Docker

Build from ai-learn repo root (Dockerfile is at root):

```bash
# From ai-learn repo root:
docker build -t mw2 .
docker run -p 8000:8000 mw2
```
