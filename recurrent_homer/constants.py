from pathlib import Path

DATA_PATH = Path("data")
WIKI_DATASET_PATH = DATA_PATH / "wikipedia_dataset"
WIKI_MODEL_PATH = DATA_PATH / "wiki_model"
HOMER_DATASET_PATH = DATA_PATH / "homer_dataset"
HOMER_MODEL_PATH = DATA_PATH / "homer_model"
APP_MODEL_PATH = DATA_PATH / "app_model"

# Modelling constants
LR_SCHEDULER_STEP_WIKI = 5
LR_SCHEDULER_STEP_FINE_TUNE = 1
