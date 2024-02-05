import streamlit as st

def show():
    image_url = "https://github.com/marko-londo/Sfere/raw/main/Images/home.jpg"
    
    # Center the image using custom CSS
    st.write(
        f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="400"></div>',
        unsafe_allow_html=True,
    )

    # Add a centered subheader
    st.write(
        '<div style="text-align: center;"><h2>NLP solutions for a Digital Earth</h2></div>',
        unsafe_allow_html=True,
    )