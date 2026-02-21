# Running Redis and PostgreSQL in the local Kubernetes cluster

The **local** Kustomize overlay deploys Redis and PostgreSQL in the `t7e` namespace. Postgres uses a custom image **t7e/postgres-pgvector:16** (Postgres 16 + pgvector via apt); you must build it once before applying the overlay (see [Building the Postgres image](#building-the-postgres-image)). Both services use **PersistentVolumeClaims** so data survives pod restarts.

**Local only:** These in-cluster services are defined only in `t7e/k8s/overlays/local`. Dev and prod overlays do not include them; cloud environments use managed Redis and Postgres (e.g. RDS, ElastiCache, Cloud SQL, managed Redis) and inject connection details via config or external secrets.

## Prerequisites

- **Docker Desktop** with Kubernetes enabled (Settings → Kubernetes → Enable).
- `kubectl` configured to use that cluster (`kubectl config current-context` should show docker-desktop).

## Cluster access vs external access

The Redis and Postgres pods use **hostPort** (6379 and 5432), so they bind directly to the node’s ports. The Services use **NodePort** so they are exposed on a port on every node. If your cluster exposes node ports to the host (e.g. Docker Desktop), use **localhost** with no port-forward:

**In-cluster:** `postgres` and `redis` are ClusterIP so other pods use `postgres:5432` and `redis:6379`. **External:** use LoadBalancer EXTERNAL-IP or port-forward:

**Port-forward** (for localhost access):

```bash
kubectl port-forward -n t7e svc/redis 6379:6379 &
kubectl port-forward -n t7e svc/postgres 5432:5432 &
```

Then use `localhost:6379` and `localhost:5432`.

**LoadBalancer:** `postgres-external` and `redis-external` get an EXTERNAL-IP when the cluster has a load balancer (e.g. MetalLB). If they stay &lt;pending&gt;, use port-forward above or remove the loadbalancer manifests from kustomization.

**Apply the overlay** (from ai-learn repo root):

```bash
kubectl apply -k t7e/k8s/overlays/local
```

## Start the services

From the **ai-learn repo root** (if the cluster already exists):

```bash
kubectl apply -k t7e/k8s/overlays/local
```

Wait until both pods are running:

```bash
kubectl get pods -n t7e -w
```

When `redis-*` and `postgres-*` show `Running`, exit with Ctrl+C.

### Stopping port-forwards

If you started port-forwards in the background, stop them with `kill %1 %2` (or kill the relevant job PIDs).

After port-forward (or LoadBalancer EXTERNAL-IP), you can connect from your machine:

- **Redis:** `redis-cli` (connects to localhost:6379). Or URL: `redis://localhost:6379/0`
- **Postgres:** `psql -h localhost -d postgres -U postgres` (password: `localdev`).

| Service   | Host (port-forward) | Port | Notes |
|-----------|---------------------|------|--------|
| Redis     | localhost           | 6379 | `redis-cli` |
| Postgres  | localhost           | 5432 | `psql -h localhost -d postgres -U postgres`, password `localdev` |

### Building the Postgres image

The overlay expects the image `t7e/postgres-pgvector:16`. Build it once (from anywhere in the repo):

```bash
t7e/k8s/scripts/build-postgres-image.sh
```

Or from the ai-learn repo root: `docker build -f t7e/docker/postgres-pgvector.Dockerfile -t t7e/postgres-pgvector:16 .`

Docker Desktop Kubernetes uses images from the local Docker daemon. For other clusters (e.g. kind), load the image after building (see `t7e/docker/README.md`). Create the pgvector extension manually if needed: `psql -h localhost -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"`

## Stop the services

1. **If you used manual port-forwards**, stop them (Ctrl+C in those terminals).

2. **Option A: Remove only the workloads (keep data)**  
   Scale down so the PVCs and data remain for next time:

   ```bash
   kubectl scale deployment -n t7e redis postgres --replicas=0
   ```

   To start again later:

   ```bash
   kubectl scale deployment -n t7e redis postgres --replicas=1
   ```

3. **Option B: Remove everything including data**  
   Delete the namespace (PVCs and data are removed):

   ```bash
   kubectl delete namespace t7e
   ```

   To start again from scratch, run `kubectl apply -k t7e/k8s/overlays/local` again.

## Useful commands

```bash
# List pods in t7e
kubectl get pods -n t7e

# Logs
kubectl logs -n t7e -l app=redis -f
kubectl logs -n t7e -l app=postgres -f

# Describe PVCs (check binding / capacity)
kubectl get pvc -n t7e
```

## Storage

- **Redis**: PVC `redis-data` (1Gi), mounted at `/data`; Redis runs with `--appendonly yes` so restarts keep data.
- **Postgres**: PVC `postgres-data` (2Gi), mounted at `/var/lib/postgresql/data`. Image `t7e/postgres-pgvector:16` (Postgres 16 + pgvector); build from `t7e/docker/` (see above).
