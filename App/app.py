import streamlit as st
from Pages import page1,page2, page3, page4, page5

st.set_page_config(
    page_title="Sfere App",
    page_icon=":crystal_ball:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)


def main():
    st.sidebar.title("Navigation")
    page_options = [
        "Home",
        "Text Processor",
        "NLP Preprocessor",
        "Network Graph",
        "ML"
        ]
    selected_page = st.sidebar.radio("Select Page", page_options)

    if selected_page == "Home":
        page1.show()
    elif selected_page == "Text Processor":
        page2.show()
    elif selected_page == "NLP Preprocessor":
        page3.show()
    elif selected_page == "Network Graph":
        page4.show()
    elif selected_page == "ML":
        page5.show()

if __name__ == "__main__":
    main()
