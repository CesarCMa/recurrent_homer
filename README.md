<img src="https://github.com/CesarCMa/recurrent_homer/blob/main/recurrent_homer/img/header.jpg" width="175" align="right" /></a>
# Recurrent Homer

This is a small project developed during the Deep Learning module within an [AI Master's degree](https://idal.uv.es/master_ia3/) at the University of Valencia.

Tha main idea of the project was to develop a simple Deep Learning model to generate text with the style of Homer Simpson.

The results are far away from impressive but, as this was my first contact with text generation models I am quite happy with the fact that the model is able to generate relatively meaningful sentences.

## Setup Of the project

If you want to test out the Streamlit app follow these steps:

1. Clone the repository on your local machine: `git clone https://github.com/CesarCMa/recurrent_homer.git`
2. Install poetry: `pip install poetry`
3. Install project requirements: `poetry install` 
4. Clone the pre-trained model files on `data/app_model` repository. You can find all required files at [this drive folder](https://drive.google.com/drive/folders/1O7Cnsm56JprPkG18n5PIGaSBzakJbCKT?usp=drive_link).
5. Run the app: `streamlit run recurrent_homer/app.py`