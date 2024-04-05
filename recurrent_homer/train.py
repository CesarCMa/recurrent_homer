"""`train` implementation module."""

import logging
from pathlib import Path

import click
import tensorflow as tf

from recurrent_homer.constants import DATA_PATH, WIKI_DATASET_PATH, WIKI_MODEL_PATH
from recurrent_homer.jobs import train_wiki_model
from recurrent_homer.model.text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


@click.command()
@click.option("--epochs", type=int, default=1, help="Number of epochs to train the model.")
@click.option(
    "--embedding_dim",
    type=int,
    default=256,
    help="Embedding dimension for the first layer of the model.",
)
@click.option(
    "--rnn_units",
    type=int,
    default=1024,
    help="Numer of recurrent units for the biggest layer of the model.",
)
@click.option(
    "--n_gru_layers", type=int, default=1, help="Number of GRU layers to use in the model."
)
@click.option("--dropout", type=float, default=0.2, help="Dropout rate to use in the model.")
@click.option(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size used during dataset creation.",
)
def train(
    epochs: int,
    embedding_dim: int,
    rnn_units: int,
    n_gru_layers: int,
    dropout: float,
    batch_size: int,
):
    """Train model."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading train and validation sets...")
    wiki_train_set, wiki_val_set = _load_train_val_dataset(WIKI_DATASET_PATH)

    logger.info("Loading Vocabulary...")
    text_vectorizer = TextVectorizer(vocabulary_path=DATA_PATH / "vocabulary.pkl")

    logger.info("Training model...")
    wiki_recurrent_model, wiki_history, wiki_val_loss = train_wiki_model(
        wiki_train_set,
        wiki_val_set,
        text_vectorizer,
        epochs,
        embedding_dim,
        rnn_units,
        n_gru_layers,
        dropout,
        batch_size,
    )

    logger.info(f"History of wiki model: {_process_model_history(wiki_history)}")
    logger.info(f"Validation loss: {wiki_val_loss}")

    logger.info("Saving wiki model...")
    wiki_recurrent_model.save_weights(WIKI_MODEL_PATH)


def _load_train_val_dataset(path: Path) -> tuple:
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


def _process_model_history(history: tf.keras.callbacks.History) -> dict:
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


if __name__ == "__main__":
    train()
