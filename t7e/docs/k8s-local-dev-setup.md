# Kubernetes local dev: web app + Postgres + Redis (WSL → cluster, Windows browser)

Goal:

- **Web app** runs in the Docker Desktop Kubernetes cluster and is visible in a **Windows browser**.
- **Postgres** and **Redis** run in the same cluster; you connect from **WSL** for local debugging (scripts, migrations, CLI tools).

## 1. Run Postgres and Redis in the cluster

Create a namespace and deploy Postgres and Redis (example: no Helm).

**Namespace**

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: app
```

**Postgres (single instance, not HA)**

```yaml
# postgres.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: app
data:
  POSTGRES_DB: myapp
  POSTGRES_USER: myapp
  POSTGRES_PASSWORD: myapp
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: app
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          ports:
            - containerPort: 5432
          envFrom:
            - configMapRef:
                name: postgres-config
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: app
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  type: ClusterIP
```

**Redis**

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: app
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
  type: ClusterIP
```

**Secret for app connections** (optional; use this if your web app expects a Secret):

```yaml
# postgres-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: app
type: Opaque
stringData:
  password: myapp
```

Apply from WSL:

```bash
kubectl apply -f namespace.yaml
kubectl apply -f postgres.yaml
kubectl apply -f redis.yaml
kubectl apply -f postgres-secret.yaml   # if using the webapp template below
```

## 2. Deploy your web app into the cluster

Your web app should:

- Connect to Postgres at `postgres.app.svc.cluster.local:5432` (or `postgres:5432` if in the same namespace).
- Connect to Redis at `redis.app.svc.cluster.local:6379` (or `redis:6379`).

Use env vars or a ConfigMap, e.g.:

- `POSTGRES_HOST=postgres` (or `postgres.app.svc.cluster.local`)
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=myapp`
- `POSTGRES_USER=myapp`
- `POSTGRES_PASSWORD=myapp`
- `REDIS_HOST=redis`
- `REDIS_PORT=6379`

Expose the app with a **Service** and **NodePort** (or LoadBalancer) so the Windows host can reach it. Example:

```yaml
# webapp.yaml (template)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
  namespace: app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
        - name: webapp
          image: your-registry/your-webapp:latest   # or imagePullPolicy: Never + local image
          env:
            - name: POSTGRES_HOST
              value: postgres
            - name: POSTGRES_PORT
              value: "5432"
            - name: POSTGRES_DB
              value: myapp
            - name: POSTGRES_USER
              value: myapp
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            - name: REDIS_HOST
              value: redis
            - name: REDIS_PORT
              value: "6379"
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: webapp
  namespace: app
spec:
  selector:
    app: webapp
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080
  type: NodePort
```

On Docker Desktop, NodePort services are usually bound to the host, so from a **Windows browser** you can open:

- `http://localhost:30080`

(If you use LoadBalancer, Docker Desktop often gives you `localhost` as well.)

## 3. Access Postgres and Redis from WSL for debugging

Cluster services are not directly routable from WSL. Use **port-forward** so that `localhost` in WSL points at the cluster Postgres and Redis:

```bash
# From WSL - run in two terminals or in background
kubectl port-forward -n app svc/postgres 5432:5432
kubectl port-forward -n app svc/redis 6379:6379
```

Then from WSL:

- **Postgres**: `psql -h localhost -p 5432 -U myapp -d myapp` (password: `myapp`)
- **Redis**: `redis-cli -h localhost -p 6379`

Your scripts, migrations, and IDE in WSL can use `localhost:5432` and `localhost:6379`; traffic is forwarded to the same instances the cluster web app uses.

Optional: run port-forwards in the background and leave them up while you work:

```bash
kubectl port-forward -n app svc/postgres 5432:5432 &
kubectl port-forward -n app svc/redis 6379:6379 &
```

## 4. Quick reference

| What              | Where to use it   | How |
|-------------------|-------------------|-----|
| Web app in browser| Windows           | `http://localhost:30080` (or your NodePort/LB port) |
| Postgres from WSL | Scripts, psql, IDE| `kubectl port-forward -n app svc/postgres 5432:5432` then `localhost:5432` |
| Redis from WSL    | redis-cli, scripts| `kubectl port-forward -n app svc/redis 6379:6379` then `localhost:6379` |

## 5. Optional: Postgres data persistence

The Postgres manifest above uses a PVC. On Docker Desktop the default storage class usually provisions a volume so data survives pod restarts. Redis uses `emptyDir` so data is lost when the pod restarts; for debugging that’s often acceptable. For Redis persistence you can add a PVC and mount it at `/data` and use a Redis config that enables RDB/AOF if needed.
