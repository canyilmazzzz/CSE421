import os

BASE_DIR = os.path.dirname(__file__)

REGRESSION_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# make sure the directory exists
os.makedirs(REGRESSION_MODEL_DIR, exist_ok=True)
