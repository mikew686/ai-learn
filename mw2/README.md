# mw2

Flask app for mw2, an AI learning and example website.

# environment

```bash
# From ai-learn repo root:
pip install -e .
```

## background services (k8s)

Can install postgres and redis locally, or use Kubernetes versions.
Assumes that a docker services with Kubernetes (like `docker desktop`) is installed:

```bash
kubectl apply -k mw2/k8s/overlays/local # start both services on k8s and provide local ports
```

## run the apps locally

After background service, can run the apps. Set up `.env` first, see `.env_example`.

```bash
# Run the app (from any directory):
python -m app

# From a different terminal run the rqworker
python -m rqworker
```

Then open http://127.0.0.1:5000/.
