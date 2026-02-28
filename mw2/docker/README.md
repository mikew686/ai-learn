# t7e Docker images

## postgres-pgvector

Postgres 16 (Debian Bookworm) with the **pgvector** extension installed via apt (`postgresql-16-pgvector` from PGDG). Used by the local Kustomize overlay so the app can use vector search.

**Build** (from anywhere in the repo):

```bash
t7e/k8s/scripts/build-postgres-image.sh
```

Or from repo root: `docker build -f t7e/docker/postgres-pgvector.Dockerfile -t t7e/postgres-pgvector:16 .`

**Use in local k8s:** The overlay `t7e/k8s/overlays/local` expects the image `t7e/postgres-pgvector:16`. After building, the image is available to Docker Desktop Kubernetes. For other clusters (e.g. kind), load it:

```bash
kind load docker-image t7e/postgres-pgvector:16 --name <your-cluster>
```

See `t7e/docs/k8s_local.md` for applying the overlay and connection details.
