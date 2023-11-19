import re
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from PIL import Image


gen_mod_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\GPT2\generator_modelv5"
gen_tok_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\GPT2\gen_tokenizerv5"
pred_mod_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\Keras\is_dying_earth_model.keras"
pred_tok_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\Keras\tokenizer.pkl"
starter_prompts_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\prompt_starters_list.pickle"
non_dying_earth_text_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\book_dictionary.pkl"
images_path= r'Pages\Artwork'

gen_model = GPT2LMHeadModel.from_pretrained(gen_mod_path)
gen_tokenizer = GPT2Tokenizer.from_pretrained(gen_tok_path)
pred_model = load_model(pred_mod_path)

with open(pred_tok_path, 'rb') as handle:
    pred_tokenizer = pickle.load(handle)

with open(starter_prompts_path, 'rb') as file:
    starter_prompts = pickle.load(file)
    
with open(non_dying_earth_text_path, 'rb') as file:
    non_dying_earth_text = pickle.load(file)
    
stop_words = set(stopwords.words('english'))

explainer = LimeTextExplainer(class_names=["Not Dying Earth", "Dying Earth"])

image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]

def clean_and_preprocess_text(text):
    tokenizer = RegexpTokenizer(r"\w[\w']+") 
    text_cleaned = re.sub(r'\d+', '', text)
    words = tokenizer.tokenize(text_cleaned)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_words)

def remove_stop_words(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess_text(text):
    text_no_stopwords = remove_stop_words(text)
    sequences = pred_tokenizer.texts_to_sequences([text_no_stopwords])
    padded_sequences = pad_sequences(sequences, maxlen=438)
    reshaped_input = padded_sequences.reshape(padded_sequences.shape[0], padded_sequences.shape[1], 1)
    return reshaped_input

def predict_dying_earth(text):
    preprocessed = preprocess_text(text)
    prediction = pred_model.predict([preprocessed])
    return prediction[0][0]

def predict_probability(texts):
    preprocessed = np.vstack([preprocess_text(text) for text in texts])
    predictions = pred_model.predict(preprocessed)
    return np.hstack((1-predictions, predictions))

def explain_prediction(text):
    exp = explainer.explain_instance(text, predict_probability, num_features=10)
    return exp.as_list()

def trim_to_last_sentence(text):
    sentence_endings = re.finditer(r'[".!?]', text)
    positions = [match.end() for match in sentence_endings]

    if positions:
        last_position = positions[-1]
        return text[:last_position].strip()
    else:
        return text
    
def text_generation(prompt):
    with st.spinner('Generating text...'):
        input_ids = gen_tokenizer.encode(prompt, return_tensors='pt')

        input_ids = input_ids.to(gen_model.device)

        max_length = len(input_ids.tolist()[0]) + 150

        output = gen_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.4,
            top_k=10,
            top_p=0.5,
            pad_token_id=gen_tokenizer.eos_token_id,
            attention_mask=input_ids.new_ones(input_ids.shape)
        )

        generated_text = gen_tokenizer.decode(output[0], skip_special_tokens=True)
        trimmed_text = trim_to_last_sentence(generated_text)
    generated_text_cleaned = clean_and_preprocess_text(trimmed_text)
    return trimmed_text, generated_text_cleaned

def image_loader():
    if image_files:
        file_path = os.path.join(images_path, random.choice(image_files))

        st.image(file_path)
    else:
        st.write("No images found in the specified directory.")
        
def display_random_text():

    random_title = random.choice(list(non_dying_earth_text.keys()))

    random_text = random.choice(non_dying_earth_text[random_title])

    return random_title, random_text

def generate_non_dying_earth_text():
    random_title, random_text = display_random_text()
    with st.spinner("Loading random book text ..."):
        st.write(random_text)
        st.write(f'Text taken from: {random_title}')
    cleaned_text = clean_and_preprocess_text(random_text)
    show_analysis(cleaned_text)
    plot_logic(random_text)

def user_input_analysis(user_input):
        with st.spinner('Analyzing text...'):
            user_input_cleaned = clean_and_preprocess_text(user_input)
        return user_input_cleaned

def correct_count():
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
    st.session_state.correct_count += 1

def show_analysis(text):
    dying_earth_score = predict_dying_earth(text)
    st.write(f"Dying Earth Probability: {dying_earth_score * 100:.2f}%")

    if dying_earth_score > 0.8:
        st.write("✔️ This text is likely from a Dying Earth novel. ")
        st.write("Consecutive Generations Matching 'Dying Earth' Style:", st.session_state.correct_count)
    else:
        st.write("❌ This text is likely not from a Dying Earth novel.")
        st.session_state.correct_count = 0

def sentence_plot(generated_text_cleaned):
    probabilities = []
    plot_sentences = []
    
    sentences = sent_tokenize(generated_text_cleaned)
    
    for sentence in sentences:
        plot_sentences.append(sentence)
        sentence_cleaned = clean_and_preprocess_text(sentence)
        
        dying_earth_score = predict_dying_earth(sentence_cleaned)
        probabilities.append(dying_earth_score)

    df = pd.DataFrame({
        'Sentence': range(1, len(sentences) + 1),
        'Probability': probabilities,
        'Text': plot_sentences
    })

    fig = px.line(df, x='Sentence', y='Probability',
                title='Probability of Each Sentence Being from a Dying Earth Novel (Stop Words Removed)',
                hover_data=['Text'])
    fig.update_traces(textposition='top center')
    fig.update_layout(width=800, height=400)
    
    return fig

def feature_plot(generated_text_cleaned):
    explanation = explain_prediction(generated_text_cleaned)

    sorted_explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=False)

    explanation_df = pd.DataFrame(sorted_explanation, columns=["Feature", "Importance"])

    fig2 = px.bar(explanation_df, x="Importance", y="Feature", orientation="h", title="Features in Order of Importance (High to Low):")
    fig2.update_layout(width=900, height=600)
    
    return fig2

def show_logic(expander_title, fig):
    with st.expander(expander_title):
        st.plotly_chart(fig) 
        
def plot_logic(generated_text_cleaned):
    st.write("Logic behind the model's predictions:")
    with st.spinner('Analyzing text...'):
        sentence_fig = sentence_plot(generated_text_cleaned)
        feature_fig = feature_plot(generated_text_cleaned) 
        show_logic("Probability by Sentence", sentence_fig)
        show_logic("Feature Importance", feature_fig) 

def show_columns(col_1, col_2):
    prompt = random.choice(starter_prompts)
    col_1, col_2 = st.columns([0.7, 0.3])
    with col_1:
        trimmed_text, generated_text_cleaned = text_generation(prompt)
        st.write(trimmed_text)
        st.write(f'Starter prompt: "{prompt}"')
        show_analysis(generated_text_cleaned)
        
    with col_2:
        image_loader()
    plot_logic(generated_text_cleaned)
       
        
def show_tab2(tab):
    with tab:
        
        st.subheader("Try it with your own text!")
        
        user_input = st.text_area("Type or paste your text here:")
        if user_input:
            correct_count()
            user_input_cleaned = user_input_analysis(user_input)
            show_analysis(user_input_cleaned)
            plot_logic(user_input_cleaned)

def show():
    
    tab1, tab2 = st.tabs(["Model Generation", "User Input"])
    col_1, col_2 = st.columns([0.7, 0.3])
    
    def show_tab1(tab):
        with tab:
            st.header("Dual-Model Text Generation and Analysis") 
            st.divider()
            st.subheader("Dying Earth Text Generator") 
            st.write('Click the "Lore Genesis" button below to generate text in the style of The Dying Earth, using a fine-tuned GPT2LMHeadModel.')
            st.write('Click the "Other Worlds" button for a random excerpt from a diverse range of books outside The Dying Earth universe.')
            st.divider()
            st.button("Cosmic Reset", help="Cast the Spell of Restoration", type="primary")
            if st.button("Lore Genesis", help="Cast the Spell of Generative Text", on_click=correct_count):
                show_columns(col_1, col_2)
            if st.button("Other Worlds", help="Summon a Codex of Otherworldly Origins"):
                generate_non_dying_earth_text()
    with tab1:
        show_tab1(tab1)

    with tab2:
        show_tab2(tab2)