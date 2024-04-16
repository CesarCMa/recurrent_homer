<img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/header.jpg" width="175" align="right" /></a>
# Recurrent Homer

This is a small project developed during the Deep Learning module within an [AI Master's degree](https://idal.uv.es/master_ia3/) at the University of Valencia.

Tha main idea of the project was to develop a simple Deep Learning model to generate text with the style of Homer Simpson.

The results are far away from impressive but, as this was my first contact with text generation models I am quite happy with the fact that the model is able to generate relatively meaningful sentences.

The model is served on a Streamlit app that lets us provide a prompt and regulate the temperature (how conservative the model should behave) and length of the output.

<img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/land_page_app.png"/></a>

## Modelling approach

<figure>
  <img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/modelling_diagram.png"/>
  <figcaption>Figure 2: Overview of the modelling approach. </figcaption>
</figure>

As you may see on the diagram, the training of the model involves two different data sources and two steps of training, lets describe it with a bit more of detail.

### Data Sources

The model was trained on subsets of two main sources:

* [Wikipedia dataset](https://huggingface.co/datasets/wikipedia) from HuggingFace
* [Dialogue Lines of The Simpsons](https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons) from Kaggle.




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

