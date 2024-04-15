import logging

import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from recurrent_homer.jobs.inference import InferenceJob


def app():

    render_header()
    with st.form("promp_submit_form", border=False):
        user_prompt = _render_prompt_input()
        submitted = st.form_submit_button("Send prompt ⬆️")
        temperature, length_response = _render_model_param_input()
        if submitted:
            with st.status("Generating response... 💭", expanded=True):
                inference_job = InferenceJob(length_response, temperature)
                model_output = inference_job.generate_response(user_prompt)
                st.write(model_output)
        else:
            _render_prompt_instructions()
        _render_temperature_instructions()


def _render_prompt_input() -> str:
    user_prompt = st.text_input(" ", placeholder="Provide your prompt here: ")
    return user_prompt


def _render_model_param_input():
    temperature = st.select_slider(
        "Select temperature of the model",
        options=[temp / 10 for temp in range(1, 20, 1)],
    )
    length_response = st.slider(
        "Select length of the response",
        min_value=1,
        max_value=300,
    )
    return temperature, length_response


def _render_prompt_instructions():
    with stylable_container(
        key="prompt_usage",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                background-color: #e6e6e6;
                opacity: 0.5;
                color: grey; 
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
    ):
        st.markdown(
            """
                **Note:** Prompt input is limited to 40 characters.
            """
        )


def _render_temperature_instructions():
    with stylable_container(
        key="temp_instructions",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                background-color: #e6e6e6;
                opacity: 0.5;
                color: black; 
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
    ):
        st.markdown(
            """
                **Temperature of the model**

                When the **temperature is low** (close to zero), the model is more conservative and
                tends to generate more deterministic outputs. This means the model is more likely
                to select tokens with higher probabilities, leading to more predictable and less
                diverse outputs.

                On the other hand, when the **temperature is high**, the model becomes more exploratory and generates more
                diverse outputs. This allows the model to explore less probable tokens and produce
                more varied and creative text. However, high temperature can also lead to more
                randomness and sometimes less coherence in the generated text.
            """
        )


def render_header():
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Recurrent Homer")
        st.subheader(':gray["Operator! Give me the number for 911!"]', divider="grey")

    with header_col2:
        st.image("recurrent_homer/img/header.jpg")

    st.markdown(
        """
        Create Your Own Homer Simpson-Style Messages!
        """
    )


if __name__ == "__main__":
    app()
