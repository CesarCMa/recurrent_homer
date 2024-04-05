"""`fine_tune_homer` module. """

import logging
import os
from pathlib import Path

import tensorflow as tf

from recurrent_homer.constants import DATA_PATH, LR_SCHEDULER_STEP_FINE_TUNE
from recurrent_homer.model.recurrent_model import RecurrentModel
from recurrent_homer.model.text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


def fine_tune_homer(
    recurrent_model: RecurrentModel,
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-9,
    checkpoint_dir: Path = DATA_PATH / "fine_tune_checkpoints",
) -> tuple:
    """Fine tune recurrent model on Homer dataset.

    Args:
        recurrent_model (RecurrentModel): Pretrained model to fine tune.
        train (tf.data.Dataset): Train set with Homer text.
        validation (tf.data.Dataset): Validation set.
        epochs (int): Number of epochs to fine tune the model.
        batch_size (int): Batch size of the datasets.
        learning_rate (float, optional): Learning rate. Defaults to 1e-9.
        checkpoint_dir (Path, optional): Directory to save checkpoints. Defaults to DATA_PATH/"training_checkpoints".

    Returns:
        tupe: Tuple with model, history and validation loss.
    """
    recurrent_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
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

    val_loss = recurrent_model.evaluate(validation, batch_size=batch_size)

    return recurrent_model, history, val_loss


def learning_rate_scheduler(epoch, lr):
    if epoch < LR_SCHEDULER_STEP_FINE_TUNE:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
