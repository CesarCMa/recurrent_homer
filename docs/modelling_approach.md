# Modelling approach

<div style="align: center; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/modelling_diagram.png"/>
  <figcaption>Figure 1: Overview of the modelling approach. </figcaption>
</div>


As you may see on the diagram, the training of the model involves two different data sources and two steps of training, lets describe it with a bit more of detail.

### Data Sources

The model was trained on subsets of two main sources:

* [Wikipedia dataset](https://huggingface.co/datasets/wikipedia) from HuggingFace
* [Dialogue Lines of The Simpsons](https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons) from Kaggle.



In the case of the Wikipedia dataset, we extract a total of 8000 samples from the original dataset, put them all together on a text corpus, and **generate sequences of 100 characters** from it.

For the Dialogue Lines we extracted the samples of the dataframe where the column *raw_character_text* was equal to *Homer Simpson*, then followed the same logic as with the wiki dataset, create a corpus and generate sequences of 100 characters.

### Preprocess

One of the key steps to build a text generation model is the **Vectorization of the text**, this is, the process of converting text into a numerical representation that our Deep Learning models can understand.

Also, we have to take into account that our main objective is to create a model that generates text, so we need to provide pairs of $(x,y)$ to achieve so.

<div style="align: center; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/preprocess_diagram.png"/>
  <figcaption>Figure 2: Overview of the preprocessing. </figcaption>
</div>

The steps to create such pairs of samples, and to vectorize the text is the following:

1. Take our text corpus and create a vocabulary (set of unique characters on the corpus). In our case, because the wikipedia dataset is the largest, we have build the vocabulary baset on it.
2. Create a `TextVectorizer` with this vocabulary, which consist of a mapper that assigns a numerical value to each character on our vocabulary.
3. Convert all our corpus to numerical values using our `TextVectorizer`.
4. Generate pairs of sequences of 100 characters (our $(x,y)$ pairs for the model), where each pair consist of a given sequence of text and the same sequence moved one character forward (Figure 4).

<div style="align: center; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/input_output.png"/>
  <figcaption>Figure 3: Sample of pair of text sequences (x,y). </figcaption>
</div>

### Model Training

As commented on the introduction to the modelling approach, the training is composed of two main steps, let's describe both in detail:

#### 1. Training of a general text generation model with Wikipedia dataset.

During the development project several architechtures were tested but the final architechture used (and the one used on the default model from the google drive folder) is composed of:

* An embedding layer which works as a trainable lookup table that will map each character-ID to a vector with embedding_dim dimensions.
* Five [GRU layers](https://en.wikipedia.org/wiki/Gated_recurrent_unit) with # of units equal to 256, 512, 1024, 512 and 256.
* A dense layer with # of units equal to length of the vocabulary. It outputs one logit for each character in the vocabulary.

<div style="align: center; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/model_arch.png"/>
  <figcaption>Figure 4: Model architecture overview. </figcaption>
</div>


The dropout of the model was set to 0.5 and the metric used to evaluate the model was [Sparse Categorical Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy).

The loss values for the final model (on the first train stage, with wikipedia dataset) where 1.47 for train and 1.42 for validation set. The model was trained during 50 epochs, on a Google Colab environment with a V100 GPU, each epoch taking around 25 minutes to complete.

#### 2. Fine tune on Homer Simpson dataset.

After training our model on the wikipedia dataset, we perform a fine tune of the model by training it again on Homer Simpson's script lines.

The most important part of this step is to provide a really low learning rate to the model, in order to avoid losing all the training performed previously. In particular we tune a learning rate scheduler to use a lr of 1,00E-05 during the first 4 epochs of training and to decrease exponentially after.

The fine tune was performed during 49 epochs, with final loss values of 1.73 and 1.67 for train and validation respectively