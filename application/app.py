import pickle
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


@st.cache_resource
def load_data():
    all_data = pd.read_csv('all_data.csv', index_col="Unnamed: 0")
    with open('bm25.pickle', 'rb') as f:
        bm25 = pickle.load(f)
    with open('id_dict.pickle', 'rb') as f:
        id_dict = pickle.load(f)
    stemmer = SnowballStemmer("english")
    russian_stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('english')) | set(stopwords.words('russian'))
    return all_data, bm25, id_dict, stemmer, russian_stemmer, stop_words

all_data, bm25, id_dict, stemmer, russian_stemmer, stop_words = load_data()


def stem_token(token):
    stem = stemmer.stem(token) if any(c.isalpha() for c in token) \
        else russian_stemmer.stem(token)
    return stem


def preprocess(doc):
    doc = doc.lower()
    doc = doc.replace('\n', ' ')
    tokens = word_tokenize(doc)
    stemmed_tokens = [stem_token(token) for token in tokens]
    result = [token for token in stemmed_tokens if token not in stop_words]

    return result


def get_documents(query, n):
    preprocessed_query = preprocess(query)
    doc_scores = bm25.get_scores(preprocessed_query)
    indices = np.argsort(doc_scores)[-n:]
    relevant_indexes = [id_dict[index] for index in reversed(indices)]
    relevant_docs = all_data[all_data['true_id'].isin(relevant_indexes)]

    return relevant_docs


with st.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input('Enter your query:', value='Как получить справку?')
    with col2:
        n = st.number_input('Enter number of documents to retrieve:', min_value=1, max_value=1000, value=5)

    submit_button = st.form_submit_button(label='Search', use_container_width=True, type='primary')


with st.spinner('Retrieving documents...'):
    relevant_docs = get_documents(query, n)

st.dataframe(relevant_docs.reset_index(drop=True))