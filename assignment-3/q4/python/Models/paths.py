import os

# Absolute path to the Models directory
BASE_DIR = os.path.dirname(__file__)

# Directory for saved Keras models
KERAS_MODEL_DIR = os.path.join(BASE_DIR, "keras_models")

# Ensure the directory exists
os.makedirs(KERAS_MODEL_DIR, exist_ok=True)
