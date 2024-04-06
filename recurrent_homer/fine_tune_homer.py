"""`train_homer` module. """

import json
import logging

import click

from recurrent_homer.constants import (
    DATA_PATH,
    HOMER_DATASET_PATH,
    HOMER_MODEL_PATH,
    WIKI_MODEL_PATH,
)
from recurrent_homer.jobs import fine_tune_homer
from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer
from recurrent_homer.utils import (
    get_input_shape,
    load_model_params,
    load_train_val_dataset,
    process_model_history,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--epochs", type=int, default=1, help="Number of epochs to fine tune the model.")
@click.option("--learning-rate", type=float, default=1e-9, help="Learning rate to use.")
def fine_tune(epochs: int, learning_rate: float):
    """Fine tune recurrent model on HOMER dataset.

    Args:
        epochs (int): Number of epochs to fine tune the model.
        learning_rate (float, optional): Learning rate. Defaults to 1e-9.
    """

    logging.basicConfig(level=logging.INFO)

    logger.info("Loading train and validation sets...")
    homer_train, homer_val = load_train_val_dataset(HOMER_DATASET_PATH)

    logger.info("Loading Vocabulary and Wiki model...")
    input_shape = get_input_shape(homer_train)
    wiki_model, model_params = _load_components(input_shape)

    logger.info("Fine-tuning model...")
    homer_model, history, val_loss = fine_tune_homer(
        recurrent_model=wiki_model,
        train=homer_train,
        validation=homer_val,
        epochs=epochs,
        batch_size=input_shape[0],
        learning_rate=learning_rate,
    )
    logger.info(f"History of fine-tuned model: {process_model_history(history)}")
    logger.info(f"Validation loss: {val_loss}")

    logger.info("Saving fine-tuned model...")
    homer_model.save_weights(HOMER_MODEL_PATH / "model")
    _save_model_params(model_params, input_shape)


def _load_components(input_shape: list) -> tuple:
    """Load Pretrained model and text vectorizer.

    Args:
        input_shape (list): Input shape of the pretrained model.

    Returns:
        tuple: Pretrained model and text vectorizer.
    """
    text_vectorizer = TextVectorizer(vocabulary_path=DATA_PATH / "vocabulary.pkl")

    model_params = load_model_params(WIKI_MODEL_PATH)
    logger.info(f"Loaded model parameters: {model_params}")

    model = RecurrentModel(
        vocab_size=len(text_vectorizer.ids_from_chars.get_vocabulary()),
        embedding_dim=model_params["embedding_dim"],
        rnn_units=model_params["rnn_units"],
        n_gru_layers=model_params["n_gru_layers"],
        dropout=model_params["dropout"],
    )

    logger.info(f"Building model with input shape: {input_shape}")
    model.build(input_shape=(input_shape[0], input_shape[1]))
    model.load_weights(WIKI_MODEL_PATH / "model")
    return model, model_params


def _save_model_params(model_params: dict, input_shape: list) -> None:
    """
    Save the model parameters to a JSON file.

    Parameters:
        model_params (dict): The model parameters.
        input_shape (list): The input shape of the model.

    Returns:
        None
    """
    model_params["input_shape"] = input_shape
    model_params_path = HOMER_MODEL_PATH / "model_params.json"
    with open(model_params_path, "w") as f:
        json.dump(model_params, f)


if __name__ == "__main__":
    fine_tune()
