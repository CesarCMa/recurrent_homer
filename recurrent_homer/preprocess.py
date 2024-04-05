"""`preprocess` implementation module."""

import logging

import click

from recurrent_homer import jobs
from recurrent_homer.constants import DATA_PATH, HOMER_DATASET_PATH, WIKI_DATASET_PATH

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--val-test-proportions",
    type=(int, int),
    default=(10, 10),
    help="Proportions for validation and test sets, remaining for train.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size to use when creating the datasets.",
)
@click.option(
    "--wiki-amount-samples",
    type=int,
    default=8000,
    help="Number of samples to extract from `wikipedia` dataset.",
)
def preprocess(
    val_test_proportions: tuple,
    batch_size: int,
    wiki_amount_samples: int,
):
    """Preprocess data to train the model.

    Args:
        val_test_proportions (tuple): Percentage of the total set to ve used as
          validation and test set.
        batch_size (int): Batch size to use.
        wiki_amount_samples (int): Number of samples to extract from `wikipedia` dataset.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing wikipedia dataset")
    wiki_train, wiki_val, wiki_test, text_vectorizer = jobs.preprocess_wiki_dataset(
        val_test_proportions, wiki_amount_samples, batch_size
    )

    logger.info("Preprocessing Homer dataset")
    homer_train, homer_val, homer_test = jobs.preprocess_homer_dataset(
        text_vectorizer, val_test_proportions, batch_size
    )

    logger.info("Saving datasets")
    wiki_train.save(str(WIKI_DATASET_PATH / "train"))
    wiki_val.save(str(WIKI_DATASET_PATH / "validation"))
    wiki_test.save(str(WIKI_DATASET_PATH / "test"))
    homer_train.save(str(HOMER_DATASET_PATH / "train"))
    homer_val.save(str(HOMER_DATASET_PATH / "validation"))
    homer_test.save(str(HOMER_DATASET_PATH / "test"))
    text_vectorizer.save_vocabulary(DATA_PATH)


if __name__ == "__main__":
    preprocess()
