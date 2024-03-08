from pathlib import Path

import tensorflow as tf

from recurrent_homer.jobs.inference import Inference
from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer

COMPONENTS_PATH = Path("data/model")


# TODO: Resolver duda: porqué en el código se emplea model() y no model.predict() para realizar las predicciones'??
def main():
    model, text_vectorizer = _load_components(COMPONENTS_PATH)
    prompt = "Hi, how you doing?"
    next_char = tf.constant([prompt])
    result = [next_char]
    states = None
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(next_char, "UTF-8")
    input_ids = text_vectorizer.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = model(inputs=input_ids, states=states, return_state=True)
    print(f"forecast: {predicted_logits}")


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


if __name__ == "__main__":
    main()
