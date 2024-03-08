import logging

import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from recurrent_homer.jobs.inference import InferenceJob


def app():

    render_header()
    with st.form("promp_submit_form", border=False):
        user_prompt = render_prompt_input()
        submitted = st.form_submit_button("Send prompt ⬆️")
        if submitted:
            with st.status("Generating response... 💭", expanded=True):
                inference_job = InferenceJob(100)
                model_output = inference_job.generate_response(user_prompt)
                st.write(model_output)
        else:
            _render_prompt_instructions()


def render_prompt_input() -> str:
    user_prompt = st.text_input(" ", placeholder="Provide your prompt here 👇️")
    return user_prompt


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
                **Prompt Usage** 
                * Prompt input is limited to 40 characters.
                * Output of the model is limited to 140 characters.
            """
        )


def render_header():
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Recurrent Homer")
        st.subheader(':gray["Operator! Give me the number for 911!"]', divider="grey")

    with header_col2:
        st.image("recurrent_homer/img/header.jpg")

    st.text(
        """
        You're welcome! I'm glad you liked it. If you need any more help
        or have another request, feel free to ask. Enjoy your D'oh!
        Generator project!
        """
    )


if __name__ == "__main__":
    app()
