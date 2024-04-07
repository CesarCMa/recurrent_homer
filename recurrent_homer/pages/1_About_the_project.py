import streamlit as st
from streamlit_extras.stylable_container import stylable_container


def about_the_project():
    _render_introduction()


def _render_introduction():
    st.markdown(
        """
            # About *Recurrent Homer* Project.

            This project was developed as part of a Deep Learning task for the master's program in
            Artificial Intelligence at the University of Valencia.

            The original idea of the project was to create a text generation model that produces
            text in the style of Homer Simpson.

            The final model used is a simple recurrent neural network composed of 7 layers: an
            initial embedding layer, 5 GRU layers, and a final dense layer to extract probabilities.

            The training process involved initially training the model on a dataset of Wikipedia
            text with XXX samples, followed by fine-tuning the model on a dataset of Homer's phrases.

            The final result is not spectacular, but I am quite satisfied with the fact that the
            model is able to produce somewhat coherent sentences (this depends on the temperature
            adjustment we make) with some references to Homer's phrases.

            Difficulties encountered during the project development:

            * Despite the simplicity of the model, training times with the mentioned Wikipedia
                dataset ranged from 10-15 hours in a Google Colab environment using a V100 GPU.
            * In the initial attempts, the model did not converge; it trained for a few epochs
                (3, 4), and the loss function quickly bounced back. This was resolved by increasing
                the batch size to 256 samples per batch.
            * To perform fine-tuning of the model on the set of Homer's phrases, it was necessary
                to manually adjust the initial learning rate to prevent the model from becoming
                unstable during training.
        """
    )


if __name__ == "__main__":
    about_the_project()
