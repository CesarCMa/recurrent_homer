"""`train_wiki_model` module."""

import logging
import os

import tensorflow as tf

from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


def train_wiki_model(
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    text_vectorizer: TextVectorizer,
    epochs: int,
    embedding_dim: int,
    rnn_units: int,
    n_gru_layers: int,
    dropout: float,
    batch_size: int,
    checkpoint_dir: str = "data/training_checkpoints",
):
    vocab_size = len(text_vectorizer.ids_from_chars.get_vocabulary())
    recurrent_model = _init_model(
        train.take(1),
        vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        n_gru_layers=n_gru_layers,
        dropout=dropout,
    )

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

    history = recurrent_model.fit(
        train,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_scheduler_callback],
        verbose=2,
    )

    logger.info(f"Training completed, evaluating model... {validation}")
    val_loss = recurrent_model.evaluate(validation, batch_size=batch_size)

    return recurrent_model, history, val_loss


def _init_model(
    train_set_sample,
    vocab_size: int,
    embedding_dim: int,
    rnn_units: int,
    n_gru_layers: int,
    dropout: float,
) -> RecurrentModel:
    """Create an instance of `RecurrentModel` and ensure it is correctly
    initialized.

    In order to check if the model is correctly initialized, we estimate the
    Sparse Categorical Crossentropy for one sample and check that the exponential
    of the mean loss is approximately equal to the vocabulary size at
    initialization

    Args:
        train_set_sample (tf.data.Dataset): Train set sample.
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        rnn_units (int): RNN units.
        n_gru_layers (int): Number of GRU layers.
        dropout (float): Dropout rate.

    Returns:
        RecurrentModel: Model instance.
    """
    exp_init_loss = 0
    while not (vocab_size - 1 <= exp_init_loss <= vocab_size + 1):
        logger.info("Initializing model...")
        recurrent_model = RecurrentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            n_gru_layers=n_gru_layers,
            dropout=dropout,
        )

        for input_example_batch, target_example_batch in train_set_sample:
            logger.info(f"Sample input batch: {input_example_batch}")
            example_batch_predictions = recurrent_model(input_example_batch)
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            init_loss = loss(target_example_batch, example_batch_predictions)
            exp_init_loss = tf.exp(init_loss).numpy()

    _log_model_info(
        example_batch_predictions,
        init_loss,
        exp_init_loss,
        recurrent_model,
    )
    recurrent_model.compile(optimizer="adam", loss=loss)
    return recurrent_model


def _log_model_info(
    example_batch_predictions: tf.Tensor,
    init_loss: float,
    exp_init_loss: float,
    recurrent_model: RecurrentModel,
) -> None:
    """Log model structure and output of the model by passing a sample throug it.

    Args:
        example_batch_predictions (tf.Tensor): Model output.
        init_loss (float): Initial loss.
        exp_init_loss (float): Exponential of the initial loss.
        recurrent_model (RecurrentModel): Model instance.

    """
    logger.info(
        f"""
        Sample of model output dimension:
        {example_batch_predictions.shape}
        # (batch_size, sequence_length, vocab_size)
        Init mean SCC loss: {init_loss}
        Exponential of the mean SCC: {exp_init_loss}
      """
    )
    recurrent_model.summary()


def learning_rate_scheduler(epoch: int, lr: float) -> float:
    """Learning rate scheduler.

    Args:
        epoch (int): Epoch number.
        lr (float): Learning rate.

    Returns:
        float: New learning rate.
    """
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
