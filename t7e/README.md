# t7e (translate)

Flask app for t7e; lives under `ai-learn/t7e`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for layout and deployment.

## Local run

Install from repo root so `app`, `utils`, and `rqworker` are on the path:

```bash
# From ai-learn repo root:
pip install -e .

# Run the app (from any directory):
python -m app
```

Then open http://127.0.0.1:5000/. The index page shows a hello-world message and the caller IP. You can also `import utils` and `from app import create_app`; packages come from `t7e/src` (defined in root `pyproject.toml`).

## Kustomize (local overlay)

Scaffolding only, no deployments:

```bash
kubectl apply -k t7e/k8s/overlays/local
```

## Docker

Build from ai-learn repo root (Dockerfile is at root):

```bash
# From ai-learn repo root:
docker build -t t7e .
docker run -p 8000:8000 t7e
```
