# Importaciones necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from predictors.rnn import predict as rnn_predict
from predictors.mlr import predict as mlr_predict
from predictors.bert import predict as bert_predict

import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid


model_promts = {
    "39c16e": "Summarize at least 3 elements of an ideal tragedy, as described by Aristotle.",
    "3b9047": "In complete sentences, summarize the structure of the ancient Egyptian system of government. How were different social classes involved in this government? Cite evidence from the text.",
    "814d6b": "Summarize how the Third Wave developed over such a short period of time and why the experiment was ended.",
    "ebad26": "Summarize the various ways the factory would use or cover up spoiled meat. Cite evidence in your answer.",
}

model_prompt_titles = ["The Third Wave", "Excerpt from The Jungle", "Egyptian Social Structure", "On Tragedy"]
model_prompt_text = [ "Background \n The Third Wave experiment", "With one member trimming beef", "Egyptian society was structured", "Chapter 13 \n As the sequel"]


model_options = ["RNN", "MLR", "BERT"]

# Clase para manejar la predicción con dos modelos
class PrediccionRequest:
    def __init__(self, text: str, prompt_id: str):
        self.text = text
        self.prompt_id = prompt_id

    def predict(self, model):
        if model == "RNN":
            return rnn_predict(self.text, model_promts[self.prompt_id])
        elif model == "MLR":
            return mlr_predict(self.text)
        elif model == "BERT":
            #return bert_predict(self.text, model_promts[self.prompt_id])  
            return [0, 0]          


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    .custom-font {
        font-family: 'Montserrat', sans-serif;
    }
            
    h1.custom-font{
        font-size: 32px;
        color: #d0ece7;
    }
    .st-emotion-cache-15hul6a {
        background-color: #138d75 !important;
        color: white !important;
        border: none;
        padding: 10px 20px !important;
        border-radius: 5px;
        font-size: 16px !important;
        font-weight: bold;
        cursor: pointer !important;
    }

    .st-emotion-cache-15hul6a {
        background-color: #117a65;
    }
            
    .result-box {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Roboto', sans-serif;
        color: #333;
        font-size: 24px;
        font-weight: bold;
    }
            
    label.st-emotion-cache-1qg05tj.e1y5xkzn3 {
        display: none;
    }
            
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(None, ["Predecir", "Acerca del dataset", "Ideas para mejorar"],
        icons=['clipboard2-data-fill', 'bar-chart', 'database'],
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#000"},
            "icon": {"font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#0e6655"},
            "nav-link-selected": {"background-color": "#16a085"},
        }
    )

if selected == "Predecir":
    st.markdown('<h1 class="custom-font">Predicción de contenido y redacción</h1>', unsafe_allow_html=True)

    st.markdown('<p class="custom-font"> Selecciona el modelo a utilizar: </p>', unsafe_allow_html=True)
    chosen_model = st.selectbox("", model_options)

    st.markdown('<p class="custom-font"> Selecciona el texto a predecir: </p>', unsafe_allow_html=True)
    input_text = st.text_area(" ")

    if chosen_model == "BERT" or chosen_model == "RNN":
        st.markdown('<p class="custom-font"> Selecciona el prompt a utilizar: </p>', unsafe_allow_html=True)
        promts = list(model_promts.values())
        prompt_selected = st.selectbox("", promts)        

    btn = st.button("Realizar predicción")

    if btn:
        if chosen_model == "MLR":
            pred = mlr_predict(input_text)
            st.write(f"Predicción de contenido: {pred[0]:.2f}%")
            st.write(f"Predicción de redacción: {pred[1]:.2f}%")
        elif chosen_model == "RNN":
            pred = rnn_predict(input_text, prompt_selected)
            st.write(f"Predicción de contenido: {pred[0]:.2f}")
            st.write(f"Predicción de redacción: {pred[1]:.2f}")
        elif chosen_model == "BERT":
            pred = bert_predict(input_text, prompt_selected)
            st.write(f"Predicción de contenido: {pred[0]:.2f}")
            st.write(f"Predicción de redacción: {pred[1]:.2f}")
elif selected == "Acerca del dataset":
    dataset = pd.read_csv("./data/Finaltrain.csv")
    st.markdown('<h1 class="custom-font">Acerca del dataset</h1>', unsafe_allow_html=True)
    st.write(dataset.head())