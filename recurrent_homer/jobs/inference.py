"""Inference Module."""

from pathlib import Path

import tensorflow as tf

from recurrent_homer.model.one_step_forecaster import OneStep
from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer

COMPONENTS_PATH = Path("data/model")


class InferenceJob:

    def __init__(self, len_response: int) -> None:
        self.len_response = len_response
        self.recurrent_model, self.text_vectorizer = _load_components(COMPONENTS_PATH)
        self.one_step_model = OneStep(self.recurrent_model, self.text_vectorizer)

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
        for n in range(self.len_response):
            next_char, states = self.one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        result = tf.strings.join(result)
        string_output = result[0].numpy().decode("utf-8")
        return string_output.capitalize()


def _load_components(model_path: Path) -> tuple:
    text_vectorizer = TextVectorizer(vocabulary_path=model_path / "vocabulary.pkl")
    model = RecurrentModel(
        vocab_size=len(text_vectorizer.ids_from_chars.get_vocabulary()),
        embedding_dim=256,
        rnn_units=1024,
        n_gru_layers=1,
        dropout=0.2,
    )
    model.build(input_shape=(100, 100))
    model.load_weights(model_path / "model_v0" / "checkpoint").expect_partial()
    return model, text_vectorizer
