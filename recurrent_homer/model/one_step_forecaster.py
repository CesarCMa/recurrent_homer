"""`OneStepForecaster` implementation module."""

import tensorflow as tf

from recurrent_homer.model.text_vectorizer import TextVectorizer


class OneStep(tf.keras.Model):
    def __init__(self, model, text_vectorizer: TextVectorizer, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = text_vectorizer.chars_from_ids
        self.ids_from_chars = text_vectorizer.ids_from_chars
        self._create_unknown_token_mask()

    @tf.function
    def generate_one_step(self, inputs, states=None):
        """
        A function that generates the next step (word) based on the input characters and
        model states.

        Args:
            inputs: The input characters to generate the next step from.
            states: The model states to consider during generation.

        Returns:
            predicted_chars: The predicted characters for the next step.
            states: The updated model states.
        """
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.ids_from_chars
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

    def _create_unknown_token_mask(self):
        """Create a mask to prevent "[UNK]" from being generated.

        '[UNK]' stands for 'unknown token', which is a common placeholder used in
        natural language processing (NLP) models for words or characters that are
        not present in the model's vocabulary.
        """
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(self.ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
