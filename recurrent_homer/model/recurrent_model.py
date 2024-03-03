"""`RecurrentModel` Implementation module."""

import tensorflow as tf


class RecurrentModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int,
        n_gru_layers: int,
        dropout: float,
    ):
        layer_unit_composition = self.generate_units_per_layer(rnn_units, n_gru_layers)
        print(layer_unit_composition)
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_layers = [
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
            )
            for rnn_units in layer_unit_composition
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        all_states = []
        for i, gru_layer in enumerate(self.gru_layers):
            if states is None:
                x, state = gru_layer(x, training=training)
            else:
                x, state = gru_layer(x, initial_state=states[i], training=training)
            all_states.append(state)

        x = self.dense(x, training=training)

        if return_state:
            return x, all_states
        else:
            return x

    def generate_units_per_layer(self, max_number: int, n_layers: int) -> list:
        """
        Generate a sequence that contains number of units per layer to generate a NN.
        Number of units is generated in a "pyramid" way that changes depending if
        `n_layers` is an even or odd number.

        - Even number: the two mid layers will contain `max_number` of units, and
        side layers will decrease to half of the units on each layer.
        - Odd number: the middle layer will contain `max_number` of units, and side
        layers units will decrease to half on each layer.

        Parameters:
            max_number (int): The maximum number of units in a layer.
            n_layers (int): The total number of layers in the NN.

        Returns:
            list: A list representing the number of units in each layer.
        """
        if n_layers % 2 == 0:
            mid_index1 = n_layers // 2 - 1
            mid_index2 = n_layers // 2
            layers = [
                max_number // 2 ** abs(i - mid_index1 - 0.5) for i in range(n_layers)
            ]
        else:
            mid_index = n_layers // 2
            layers = [max_number // 2 ** abs(i - mid_index) for i in range(n_layers)]
        return layers
