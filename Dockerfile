# Build from project root: docker build .
# Gunicorn runs with CWD = project root so paths and config resolve correctly.

FROM python:3.11-slim

WORKDIR /app

# Copy repo (context is ai-learn root)
COPY . .

# Install repo (includes t7e packages app, rqworker, utils from t7e/src)
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Run from project root; app is installed as package "app"
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:create_app"]
