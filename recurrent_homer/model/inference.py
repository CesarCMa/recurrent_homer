"""Inference Module."""

from pathlib import Path

import tensorflow as tf
from recurrent_homer.model.text_vectorizer import TextVectorizer
from model.one_step_forecaster import OneStep
from model.recurrent_model import RecurrentModel

COMPONENTS_PATH = Path("data/model")


def inference(
    prompt: str,
    len_response: int,
) -> str:
    recurrent_model, text_vectorizer = _load_components(COMPONENTS_PATH)
    one_step_model = OneStep(recurrent_model, text_vectorizer)
    next_char = tf.constant([prompt])
    result = [next_char]
    states = None
    for n in range(len_response):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)
    result = tf.strings.join(result)
    return result[0].numpy().decode("utf-8")


def _load_components(model_path: Path) -> tuple:
    return (
        tf.keras.models.load_model(model_path / "test_model"),
        TextVectorizer(vocabulary_path=model_path / "vocabulary.pkl"),
    )
