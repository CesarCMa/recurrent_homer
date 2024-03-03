import pandas as pd
import streamlit as st

from model.inference import inference


def app():

    render_header()
    user_prompt = render_prompt_input()
    model_output = inference(user_prompt, 140)
    st.write(model_output)


def render_prompt_input() -> str:

    user_prompt = st.text_input(" ", placeholder="Provide your prompt here 👇️")
    st.markdown(
        """
            **Prompt Usage** 
            * Prompt input is limited to 40 characters.
            * Output of the model is limited to 140 characters.
        """
    )
    return user_prompt


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
