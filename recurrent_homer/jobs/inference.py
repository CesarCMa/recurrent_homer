"""Inference Module."""

import logging
from pathlib import Path

import tensorflow as tf

from recurrent_homer.constants import APP_MODEL_PATH
from recurrent_homer.model.one_step_forecaster import OneStep
from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer
from recurrent_homer.utils import load_model_params

logger = logging.getLogger(__name__)


class InferenceJob:
    """Inference Job."""

    def __init__(self, len_response: int, temperature: float) -> None:
        """Initialize Inference Job.

        Args:
            len_response (int): Length of response.
            temperature (float): Temperature of the model, lower values generates more conservative
                responses.

        Returns:
            None
        """
        self.len_response = len_response
        self.temperature = temperature
        self.recurrent_model, self.text_vectorizer = _load_components()
        self.one_step_model = OneStep(self.recurrent_model, self.text_vectorizer, self.temperature)

    def generate_response(self, prompt: str):
        """Generate response with pretrained `RecurrentModel` instance.

        Args:
            prompt (str): User prompt.

        Returns:
            _type_: Model response.
        """
        next_char = tf.constant([prompt])
        result = []
        states = None
        logger.info("Generating response...")
        for n in range(self.len_response):
            next_char, states = self.one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        result = tf.strings.join(result)
        string_output = result[0].numpy().decode("utf-8")
        return string_output.capitalize()


def _load_components() -> tuple:
    """Load Pretrained model and text vectorizer.

    Returns:
        tuple: Pretrained model and text vectorizer.
    """
    text_vectorizer = TextVectorizer(vocabulary_path=APP_MODEL_PATH / "vocabulary.pkl")

    model_params = load_model_params(APP_MODEL_PATH)
    logger.info(f"Loaded model parameters: {model_params}")

    model = RecurrentModel(
        vocab_size=len(text_vectorizer.ids_from_chars.get_vocabulary()),
        embedding_dim=model_params["embedding_dim"],
        rnn_units=model_params["rnn_units"],
        n_gru_layers=model_params["n_gru_layers"],
        dropout=model_params["dropout"],
    )

    input_shape = model_params["input_shape"]
    logger.info(f"Building model with input shape: {input_shape}")
    model.build(input_shape=(input_shape[0], input_shape[1]))
    model.load_weights(APP_MODEL_PATH / "wiki_model")
    return model, text_vectorizer
