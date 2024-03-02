import pandas as pd
import streamlit as st


def app():

    render_header()
    st.text(
        """
        You're welcome! I'm glad you liked it. If you need any more help
        or have another request, feel free to ask. Enjoy your D'oh!
        Generator project!
        """
    )


def render_header():
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Recurrent Homer")
        st.subheader(':gray["Operator! Give me the number for 911!"]', divider="grey")

    with header_col2:
        st.image("recurrent_homer/img/header.jpg")


if __name__ == "__main__":
    app()
