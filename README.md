# Recurrent Homer

This is a small project developed during the Deep Learning module within an [AI Master's degree](https://idal.uv.es/master_ia3/) at the University of Valencia.

Tha main idea of the project was to develop a simple Deep Learning model to generate text with the style of Homer Simpson.

The results are far away from impressive but, as this was my first contact with text generation models I am quite happy with the fact that the model is able to generate relatively meaningful sentences.

The model is served on a Streamlit app that lets us provide a prompt and regulate the temperature (how conservative the model should behave) and length of the output.

<div style="align: left; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/land_page_app.png"/>
  <figcaption>Figure 1: Streamlit app.</figcaption>
</div>

## Modelling approach

<div style="align: left; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/modelling_diagram.png"/>
  <figcaption>Figure 2: Overview of the modelling approach. </figcaption>
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

The steps to create such pairs of samples, and to vectorize the text is the following:

1. Take our text corpus and create a vocabulary (set of unique characters on the corpus). In our case, because the wikipedia dataset is the largest, we have build the vocabulary baset on it.
2. Create a `TextVectorizer` with this vocabulary, which consist of a mapper that assigns a numerical value to each character on our vocabulary.
3. Convert all our corpus to numerical values using our `TextVectorizer`.
4. Generate pairs of sequences of 100 characters (our $(x,y)$ pairs for the model), where each pair consist of a given sequence of text and the same sequence moved one character forward (Figure 4).

<div style="align: left; text-align:center;">
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/preprocess_diagram.png"/>
  <figcaption>Figure 3: Overview of the preprocessing. </figcaption>
</div>

## Setup Of the Streamlit app

If you want to test out the Streamlit app follow these steps:

1. Clone the repository on your local machine: `git clone https://github.com/CesarCMa/recurrent_homer.git`
2. Install poetry: `pip install poetry`
3. Install project requirements: `poetry install` 
4. Clone the pre-trained model files on `data/app_model` repository. You can find all required files at [this drive folder](https://drive.google.com/drive/folders/1O7Cnsm56JprPkG18n5PIGaSBzakJbCKT?usp=drive_link).
5. Run the app: `streamlit run recurrent_homer/app.py`

After this steps you should be able to open the app in your browser on the localhost provided by streamlit.


## Training of the models

If you want to train you own models there are scripts fore preprocessing the data and training the models.

