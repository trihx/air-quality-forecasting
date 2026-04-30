import streamlit as st

st.set_page_config(page_title="Theme Test", layout="wide")

st.title("Theme Toggle Test")

light_mode = st.toggle("☀️ Light Mode")

if light_mode:
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
            color: #31333F !important;
        }
        [data-testid="stSidebar"] {
            background-color: #F0F2F6 !important;
        }
        [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
        }
        /* Override all text colors */
        .stMarkdown, .stText, p, span, div {
            color: #31333F;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # default dark
    pass

st.write("Hello World!")
