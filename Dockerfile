FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the latest registered model from DagsHub through MLflow at build time.
# Credentials are passed as BuildKit secrets, so they are never stored in the image.
ARG MLFLOW_TRACKING_URI=https://dagshub.com/DorDNS/mlflow-with-daghub.mlflow
ARG MODEL_NAME=digits_random_forest
# Changing this value invalidates the Docker cache so the latest model is fetched again.
ARG CACHEBUST=0

COPY scripts/ scripts/
RUN --mount=type=secret,id=mlflow_username \
    --mount=type=secret,id=mlflow_password \
    MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    MODEL_NAME="${MODEL_NAME}" \
    MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/mlflow_username)" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/mlflow_password)" \
    python scripts/download_model.py

COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
