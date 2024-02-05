import streamlit as st
import pandas as pd
from io import StringIO
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import re
import time
import pickle
import plotly.express as px
from spacy_streamlit import visualize_parser
from spacytextblob.spacytextblob import SpacyTextBlob

models = ["en_core_web_sm", "en_core_web_md"]

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

def init_session_states(**kwargs):
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_text(text):
    with st.expander("Preview"):
        st.text(text)


def to_pkl(obj):
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, TypeError) as e:
        raise Exception(f"Error pickling object: {str(e)}")


def clean_text(text, progress_callback=None):
    text = re.sub(r"\s+", " ", text).strip()
    additional_stopwords = {"said"}
    for word in additional_stopwords:
        nlp.Defaults.stop_words.add(word)
    doc = nlp(text)

    total_tokens = len(doc)
    processed_tokens = 0
    filtered_tokens = []

    for token in doc:
        if token.text.lower() not in STOP_WORDS and not token.is_punct:
            filtered_tokens.append(token.lemma_)

        processed_tokens += 1
        if progress_callback is not None:
            progress_callback(processed_tokens / total_tokens)

    return filtered_tokens, doc


def key_phrase_extraction(tokens, n=1):
    n_grams = zip(*[tokens[i:] for i in range(n)])
    n_grams = [" ".join(ngram) for ngram in n_grams]
    frequency = Counter(n_grams)
    N = 20
    key_phrases = frequency.most_common(N)
    df = pd.DataFrame(key_phrases, columns=["Phrase", "Count"])

    return df


def update_progress(progress_bar, progress):
    progress_bar.progress(progress)


def convert_df(df):
    return df.to_csv().encode("utf-8")


def ner(doc):
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    df = pd.DataFrame(named_entities, columns=["Entity", "Label"])
    return df




def file_uploader():
    init_session_states(key_phrases_df=None, n_grams=1, cleaned_text=None, pickled_tokens=None, original_text=None,
                        uploaded_file_name=None, text_cleaned=False, doc=None)

    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.original_text = uploaded_file.read().decode("utf-8")
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.text_cleaned = False

        text = st.session_state.original_text

        on = st.toggle(
            "Clean text",
            help="Processes text by tokenizing and lemmatizing, while also eliminating stop words and punctuation.",
        )

        if on and not st.session_state.text_cleaned:
            progress_bar = st.progress(0)
            st.session_state.cleaned_text, st.session_state.doc = clean_text(
                text, lambda progress: update_progress(progress_bar, progress)
            )
            st.session_state.tokens = st.session_state.cleaned_text
            st.session_state.pickled_tokens = to_pkl(st.session_state.tokens)
            st.session_state.text_cleaned = True
            progress_bar.empty()


        tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["Text Preview", "Tokens", "Key Phrases", "Named Entity Recognition", "Dependency Parsing", "Sentiment Analysis"])

        with tab1:
            show_text(text)

        with tab2:
            if st.checkbox("Download Pickled Tokens"):
                if st.session_state.pickled_tokens:
                    show_text(st.session_state.tokens)
                    st.download_button(
                        label="Download Pickled Tokens",
                        help="Download the cleaned tokens list as a .pkl file",
                        data=st.session_state.pickled_tokens,
                        file_name="tokens.pkl",
                        mime="application/octet-stream",
                    )
                else:
                    st.error("Text must be cleaned before downloading! 🚫")

        with tab3:
            if st.checkbox("Extract Key Phrases"):
                if st.session_state.cleaned_text:
                    col1, col2 = st.columns([0.25, 0.75])  # Create two columns

                    with col1:
                        st.session_state.n_grams = st.radio(
                            "Select Number of N-Grams", [3, 2, 1]
                        )

                        if st.button("Refresh", help="Updates the Key Phrases"):
                            st.session_state.key_phrases_df = key_phrase_extraction(
                                st.session_state.cleaned_text, st.session_state.n_grams
                            )

                        if st.session_state.key_phrases_df is not None:
                            # Display the dataframe
                            st.dataframe(st.session_state.key_phrases_df,
                                          height=400,
                                          width=300,
                                          hide_index=True
                                          )

                    with col2:
                        if st.session_state.key_phrases_df is not None:
                            fig = px.bar(
                                st.session_state.key_phrases_df,
                                x="Phrase",
                                y="Count",
                                title="Top 20 Key Prases"
                            )
                            fig.update_layout(width=800, height=650)
                            st.plotly_chart(fig)

                else:
                    st.error("Text must be cleaned before extracting key phrases! 🚫")

        with tab4:
            if st.checkbox("Perform Named Entity Recognition"):
                if st.session_state.doc:
                    named_entities = ner(st.session_state.doc)
                    no_duplicates = named_entities.drop_duplicates()

                    col3, col4 = st.columns([0.25, 0.75])  # Create two columns

                    with col3:
                        st.dataframe(no_duplicates,
                                    height=400,
                                    width=300,
                                    hide_index=True
                                    )

                    with col4:
                        if not named_entities.empty:
                            fig = px.bar(
                                named_entities,
                                x="Label",
                                title="Entity Type Counts"
                            )
                            fig.update_layout(width=800, height=400)
                            st.plotly_chart(fig)
                else:
                    st.error("Text must be cleaned before performing Named Entity Recognition! 🚫")


        with tab5:
            if st.checkbox("Perform Dependency Parsing"):
                if st.session_state.doc:
                    sentence_index = st.number_input("Enter sentence index:", 0, len(list(st.session_state.doc.sents))-1, 0, key="sentence_index")
                    
                    if st.button("Visualize Selected Sentence"):
                        selected_sentence = list(st.session_state.doc.sents)[sentence_index]
                        visualize_parser(selected_sentence)
                else:
                    st.error("Text must be cleaned before performing Dependency Parsing! 🚫")
                    
        with tab6:
            if st.checkbox("Perform Sentiment Analysis"):
                if st.session_state.doc:
                    col5, col6 = st.columns([0.25, 0.75])  # Create two columns

                    with col5:
                        try:
                            # Extract sentiment analysis
                            polarity = st.session_state.doc._.blob.polarity
                            subjectivity = st.session_state.doc._.blob.subjectivity

                            st.subheader("Sentiment Analysis Results:")
                            st.write(f"Polarity: {polarity}")
                            st.write(f"Subjectivity: {subjectivity}")
                        except AttributeError as e:
                            st.error(f"Error extracting sentiment analysis: {str(e)}")

                    with col6:
                        try:
                            # Extract sentiment for each sentence
                            sentiment_data = [(sentence.text, sentence._.blob.polarity, sentence._.blob.subjectivity) for sentence in st.session_state.doc.sents]

                            # Create DataFrame for display
                            if sentiment_data:
                                df_sentiment = pd.DataFrame(sentiment_data, columns=["Sentence", "Polarity", "Subjectivity"])
                                st.dataframe(df_sentiment, height=400, width=600)
                            else:
                                st.info("No sentiment assessments available.")
                        except AttributeError as e:
                            st.error(f"Error displaying sentiment assessments: {str(e)}")

                    st.divider()
                    
                    fig = px.scatter(df_sentiment, x='Polarity', y='Subjectivity',
                                     title='Sentiment Analysis: Polarity vs Subjectivity', hover_data=['Sentence'])
                    fig2 = px.line(df_sentiment, y='Polarity', title='Sentiment Polarity Over Sentences', hover_data=['Sentence'])

                    
                    fig2.update_layout(
                        xaxis_title='Sentence Index',
                        yaxis_title='Polarity',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True),
                    )                    
                    
                    with st.expander("Polarity vs Subjectivity"):
                        st.plotly_chart(fig)
                    with st.expander("Sentiment Polarity Over Sentences"):
                        st.plotly_chart(fig2)


    else:
        st.write("Please upload a .txt file")


def show():
    file_uploader()

