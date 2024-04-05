"""`train_wiki` implementation module."""

import logging
from pathlib import Path

import click
import tensorflow as tf

from recurrent_homer.constants import DATA_PATH, WIKI_DATASET_PATH, WIKI_MODEL_PATH
from recurrent_homer.jobs import train_recurrent_model
from recurrent_homer.model.text_vectorizer import TextVectorizer

from .utils import load_train_val_dataset, process_model_history

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
def train_wiki(
    epochs: int,
    embedding_dim: int,
    rnn_units: int,
    n_gru_layers: int,
    dropout: float,
    batch_size: int,
):
    """Train a recurrent model on `wikipedia` dataset. Module can be executed via CLI with command
    `python3 recurrent_homer.train_wiki.py --arg value`.

    Args:
        epochs (int): Number of epochs to train the model.
        embedding_dim (int): Embedding dimension for the first layer of the model.
        rnn_units (int): Numer of recurrent units for the biggest layer of the model.
        n_gru_layers (int): Number of GRU layers to use in the model.
        dropout (float): Dropout rate to use in the model.
        batch_size (int): Batch size used during dataset creation.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading train and validation sets...")
    wiki_train_set, wiki_val_set = load_train_val_dataset(WIKI_DATASET_PATH)

    logger.info("Loading Vocabulary...")
    text_vectorizer = TextVectorizer(vocabulary_path=DATA_PATH / "vocabulary.pkl")

    logger.info("Training model...")
    wiki_recurrent_model, wiki_history, wiki_val_loss = train_recurrent_model(
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

    logger.info(f"History of wiki model: {process_model_history(wiki_history)}")
    logger.info(f"Validation loss: {wiki_val_loss}")

    logger.info("Saving wiki model...")
    wiki_recurrent_model.save_weights(WIKI_MODEL_PATH)


if __name__ == "__main__":
    train_wiki()
