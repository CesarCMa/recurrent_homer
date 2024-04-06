import json
import logging
from pathlib import Path

import tensorflow as tf

from recurrent_homer.constants import WIKI_MODEL_PATH

logger = logging.getLogger(__name__)


def load_train_val_dataset(path: Path) -> tuple:
    """Load train and validation sets.

    Args:
        path (Path): Path to train and validation sets.

    Returns:
        tuple: Train and validation sets.
    """
    train_set = tf.data.Dataset.load(str(path / "train"))
    logger.info(f"Train set cardinality: {train_set.cardinality()}")
    validation = tf.data.Dataset.load(str(path / "validation"))
    logger.info(f"Validation set cardinality: {validation.cardinality()}")
    return train_set, validation


def process_model_history(history: tf.keras.callbacks.History) -> dict:
    """Process model history.

    Args:
        history (tf.keras.callbacks.History): History of the model.

    Returns:
        dict: Dictionary with the history of the model.
    """
    return {
        "loss": history.history["loss"],
        "learning_rate": history.history["lr"],
    }


def load_model_params(model_path: Path) -> dict:
    """
    Load model parameters from a JSON file and return them as a dictionary.
    """
    with open(model_path / "model_params.json", "r") as file:
        model_params = json.load(file)
    return model_params


def get_input_shape(train_set: tf.data.Dataset) -> list:
    """Get input shape to build the model from a give training set.

    Args:
        train_set (tf.data.Dataset): Training set.

    Returns:
        list: Input shape.
    """
    return tf.shape(list(train_set.take(1))).numpy().tolist()[-2:]
