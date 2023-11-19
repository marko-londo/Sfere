import streamlit as st
from Pages import page4

st.set_page_config(
    page_title="Dying Earth Dashboard",
    page_icon=":crystal_ball:",
    layout="wide",
    initial_sidebar_state="auto",
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
        # "Interactive Network Graph",
        "Text Analyzer",       
]
    selected_page = st.sidebar.radio("Select Page", page_options)

    # if selected_page == "Interactive Network Graph":
    #     page1.show()
    if selected_page == "Text Analyzer":
        page4.show()


if __name__ == "__main__":
    main()
