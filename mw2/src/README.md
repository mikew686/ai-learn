# t7e src

Application packages: `app` (Flask web), `rqworker` (RQ worker), `utils` (shared helpers). After `pip install -e .` from the ai-learn repo root, `t7e/src` is on the path: use `import utils`, `from app import create_app`, and run with `python -m app`.

---

## Running the Flask app

Install once from repo root: `pip install -e .`. Then:

```bash
python -m app
```

This starts the Flask dev server (debug on) at http://0.0.0.0:5000.

---

## Running the RQ worker

With Redis running:

```bash
python -m rqworker
```

Ensure Redis is available (e.g. `REDIS_URL=redis://localhost:6379/0`). The `rqworker` package is currently a stub; implement its entrypoint to start the worker (e.g. `rq worker` with the app’s worker. Alternative: `rq worker --url redis://localhost:6379/0` with optional queue names.
