import os

import mlflow
from mlflow import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "digits_random_forest")
OUTPUT_DIR = os.environ.get("MODEL_DIR", "model")

# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# are read from the environment. Never hardcode credentials here.
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise SystemExit(f"No registered model found with name '{MODEL_NAME}'")

latest = max(versions, key=lambda v: int(v.version))
print(f"Latest version of {MODEL_NAME}: {latest.version}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{MODEL_NAME}/{latest.version}",
    dst_path=OUTPUT_DIR,
)
print(f"Model downloaded to {OUTPUT_DIR}")
