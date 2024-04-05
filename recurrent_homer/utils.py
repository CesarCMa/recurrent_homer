import logging
from pathlib import Path

import tensorflow as tf

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
