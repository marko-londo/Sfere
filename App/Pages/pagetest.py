import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import plotly.express as px
import re
import random
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch



mod_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\GPT2\generator_modelv3"
tok_path = r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\GPT2\gen_tokenizerv3"

seed = 42

state = random.getstate()
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


gen_model, gen_tokenizer = load_model_and_tokenizer(mod_path, tok_path)

random.setstate(state)

with open(r'C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\prompt_starters_list.pickle', 'rb') as file:
    starter_prompts = pickle.load(file)

def clean_and_preprocess_text(text):
    text_cleaned = re.sub(r'\d+', '', text)
    words = word_tokenize(text_cleaned)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess_text_for_prediction(text):
    sequences = pred_tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=438)
    reshaped_input = padded_sequences.reshape(padded_sequences.shape[0], padded_sequences.shape[1], 1)
    return reshaped_input

def predict_dying_earth(text):
    preprocessed = preprocess_text_for_prediction(text)
    prediction = pred_model.predict([preprocessed])
    return prediction[0][0]

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

pred_model = load_model(r"C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\Keras\is_dying_earth_model.keras")
with open(r'C:\Users\dontb\01\001\Repos\Dying-Earth\Notebooks\04-Machine-Learning\Models\Keras\tokenizer.pkl', 'rb') as handle:
    pred_tokenizer = pickle.load(handle)

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

explainer = LimeTextExplainer(class_names=["Not Dying Earth", "Dying Earth"])

def explain_prediction(text):
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_list()


def predict_proba(texts):
    preprocessed = np.vstack([preprocess_text(text) for text in texts])
    predictions = pred_model.predict(preprocessed)
    return np.hstack((1-predictions, predictions))

def trim_to_last_sentence(text):
    sentence_endings = re.finditer(r"[.!?]", text)
    positions = [match.end() for match in sentence_endings]

    if positions:
        last_position = positions[-1]
        return text[:last_position].strip()
    else:
        return text
    
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0

def correct_count():
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

    st.write("Logic behind the model's predictions:")
    
    show_explanation("Probability by Sentence", fig)
    show_explanation("Features by Importance", fig2)
    
def plot_logic():
    explanation = explain_prediction(generated_text_cleaned)

    sorted_explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=False)


    explanation_df = pd.DataFrame(sorted_explanation, columns=["Feature", "Importance"])

    fig2 = px.bar(explanation_df, x="Importance", y="Feature", orientation="h", title="Features in Order of Importance (High to Low):")
    fig2.update_layout(width=900, height=600)
    
    sentences = sent_tokenize(generated_text_cleaned)

    probabilities = []
    cleaned_sentences = []

    for sentence in sentences:
        sentence_cleaned = clean_and_preprocess_text(sentence)
        

        dying_earth_score = predict_dying_earth(sentence_cleaned)
        probabilities.append(dying_earth_score)
        cleaned_sentences.append(sentence_cleaned)

    df = pd.DataFrame({
        'Sentence': range(1, len(sentences) + 1),
        'Probability': probabilities,
        'Text': cleaned_sentences
    })

    fig = px.line(df, x='Sentence', y='Probability',
                title='Probability of Each Sentence Being from a Dying Earth Novel',
                hover_data=['Text'])
    fig.update_traces(textposition='top center')
    fig.update_layout(width=900, height=600)
            
    return fig, fig2


def show_explanation(expander_title, fig):
    with st.expander(expander_title):
        st.plotly_chart(fig)

def show():
    
    tab1, tab2 = st.tabs(["Model Generation", "User Input"])
    with tab1:
        st.header("Dual-Model Text Generation and Analysis") 
        st.markdown("---")
        st.subheader("Dying Earth Text Generator") 
        st.write("Click the button below to generate text in the style of The Dying Earth, using a fine-tuned GPT2LMHeadModel.")
        
        if st.button('Lore Genesis', help="Cast the Spell of Generative Text", on_click=correct_count):
            with st.spinner('Generating text...'):
                prompt = random.choice(starter_prompts)
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
            st.write(trimmed_text)
            st.write(f'Starter prompt: "{prompt}"')

            generated_text_cleaned = clean_and_preprocess_text(generated_text)

            dying_earth_score = predict_dying_earth(generated_text_cleaned)
            st.write(f"Dying Earth Probability: {dying_earth_score * 100:.2f}%")

            if dying_earth_score > 0.8:
                st.write("✔️ This text is likely from a Dying Earth novel. ")
                st.write("Consecutive Generations Matching 'Dying Earth' Style:", st.session_state.correct_count)
            else:
                st.write("❌ This text is likely not from a Dying Earth novel.")
                st.session_state.correct_count = 0

            st.write("Logic behind the model's predictions:")
            
            with st.expander("Probability by Sentence"):
                st.plotly_chart(fig)
                
            with st.expander("Feature Importance"):
                st.plotly_chart(fig2)
                
            
    with tab2:
        
        st.subheader("Try it with your own text!")

        user_input = st.text_area("Type or paste your text here:")

        if user_input:
            with st.spinner('Analyzing text...'):
                user_input_cleaned = clean_and_preprocess_text(user_input)
                dying_earth_score = predict_dying_earth(user_input_cleaned)
                dying_earth_score = predict_dying_earth(user_input_cleaned)
                st.write(f"Dying Earth Probability: {dying_earth_score * 100:.2f}%")

                if dying_earth_score > 0.8:
                    st.write("✔️ This text is likely from a Dying Earth novel. ")
                else:
                    st.write("❌ This text is likely not from a Dying Earth novel.")

            sentences = sent_tokenize(user_input)

            probabilities = []
            cleaned_sentences = []

            for sentence in sentences:
                sentence_cleaned = clean_and_preprocess_text(sentence)
                

                dying_earth_score = predict_dying_earth(sentence_cleaned)
                probabilities.append(dying_earth_score)
                cleaned_sentences.append(sentence_cleaned)

            df = pd.DataFrame({
                'Sentence': range(1, len(sentences) + 1),
                'Probability': probabilities,
                'Text': cleaned_sentences
            })

            fig = px.line(df, x='Sentence', y='Probability',
                        title='Probability of Each Sentence Being from a Dying Earth Novel',
                        hover_data=['Text'])
            fig.update_traces(textposition='top center')
            fig.update_layout(width=900, height=600)
            
                
            explanation = explain_prediction(user_input)

            sorted_explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=False)


            explanation_df = pd.DataFrame(sorted_explanation, columns=["Feature", "Importance"])

            fig2 = px.bar(explanation_df, x="Importance", y="Feature", orientation="h", title="Features in Order of Importance (High to Low):")
            fig2.update_layout(width=900, height=600)
                
            st.write("Logic behind the model's predictions:")
            
            with st.expander("Probability by Sentence"):
                st.plotly_chart(fig)
                
            with st.expander("Feature Importance"):
                st.plotly_chart(fig2)
            