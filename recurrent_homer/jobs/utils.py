"""`utils` module for jobs. """

import logging

import tensorflow as tf
from recurrent_homer.model.text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


def create_tf_dataset(
    text: str, text_vectorizer: TextVectorizer, sequence_length: int = 100
) -> tf.data.Dataset:
    """Convert text set to `tf.data.Dataset` composed of input and target pairs.

    Args:
        text (str): Text corpus.
        text_vectorizer (TextVectorizer): Text vectorizer.
        sequence_length (int, optional): Sequence length. Defaults to 100.

    Returns:
        tf.data.Dataset: Dataset with input and target pairs.
    """
    all_ids = text_vectorizer.convert_text_to_id(text)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(sequence_length + 1, drop_remainder=True)
    return sequences.map(_split_input_target)


def _split_input_target(sequence: str):
    """Generates input and taget for the text generation model based on a
    provided sequence of ids.

    Text model is trained to predict next charachter of a sequence, for example
    if our sequence is the word "hello", input of the model will be "hell" and
    target will be "ello"
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def shuffle_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = 100,
) -> tf.data.Dataset:
    """
    Shuffle the dataset

    Args:
        dataset (tf.data.Dataset): Dataset to shuffle.
        buffer_size: Buffer size to shuffle the dataset.
        (TF data is designed to work with possibly infinite sequences,
        so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        it maintains a buffer in which it shuffles elements)

    Returns:
        tf.data.Dataset: Shuffled dataset.
    """
    buffer_size = dataset.cardinality().numpy()
    return (
        dataset.shuffle(buffer_size, reshuffle_each_iteration=False)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def train_val_test_split(full_dataset: tf.data.Dataset, val_test_prop: tuple) -> tuple:
    """Split the dataset into train, validation and test sets.

    Args:
      full_dataset (tf.data.Dataset): Dataset to split.
      val_test_prop (tuple): Proportion to use for validation and test sets.

    Returns:
      tuple: Train, validation and test sets.
    """
    logger.info("Splitting into train, val, test...")
    set_size = full_dataset.cardinality().numpy()
    train_size = int((100 - sum(val_test_prop)) / 100 * set_size)
    val_size = int(val_test_prop[0] / 100 * set_size)
    test_size = int(val_test_prop[1] / 100 * set_size)

    train = full_dataset.take(train_size)
    val_test = full_dataset.skip(train_size)
    validation = val_test.take(val_size)
    test = val_test.skip(test_size)
    _log_sets_size(set_size, train, validation, test)
    return train, validation, test


def _log_sets_size(
    set_size: int,
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    test: tf.data.Dataset,
) -> None:
    """Log size of the dataset."""
    logger.info(
        f"""
      Full dataset size: {set_size},
      train size: {train.cardinality().numpy()}
      validation size: {validation.cardinality().numpy()}
      test size: {test.cardinality().numpy()}
      """
    )
