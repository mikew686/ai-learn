# mw2 Docker images

## mw2 (app / rqworker)

**Build** (from anywhere in the repo):

```bash
mw2/scripts/build-mw2.sh
```

Or from ai-learn repo root: `docker build -f mw2/docker/mw2.Dockerfile -t mw2:latest .`

**Run the app (gunicorn) by hand:**

```bash
docker run --rm -p 8000:8000 mw2:latest
```

**Run the rqworker by hand** (override CMD; no need to clear entrypoint):

```bash
docker run --rm mw2:latest python -m rqworker
```

---

## postgres-pgvector

Postgres 16 (Debian Bookworm) with the **pgvector** extension installed via apt (`postgresql-16-pgvector` from PGDG). Used by the local Kustomize overlay so the app can use vector search.

**Build** (from anywhere in the repo):

```bash
mw2/scripts/build-postgres-image.sh
```

Or from repo root: `docker build -f mw2/docker/postgres-pgvector.Dockerfile -t mw2/postgres-pgvector:16 .`

**Use in local k8s:** The overlay `mw2/k8s/overlays/local` expects the image `mw2/postgres-pgvector:16`. After building, the image is available to Docker Desktop Kubernetes. For other clusters (e.g. kind), load it:

```bash
kind load docker-image mw2/postgres-pgvector:16 --name <your-cluster>
```

See `mw2/docs/k8s_local.md` for applying the overlay and connection details.
