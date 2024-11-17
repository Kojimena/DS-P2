# Importaciones necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from predictors.rnn import predict as rnn_predict
from predictors.mlr import predict as mlr_predict
from predictors.bert import predict as bert_predict
from predictors.svm import predict as svm_predict

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


model_options = ["RNN", "MLR", "BERT", "SVM", "Todos"]

# Clase para manejar la predicción con dos modelos
class PrediccionRequest:
    def __init__(self, text: str, prompt_id: str):
        self.text = text
        self.prompt_id = prompt_id

    def predict(self, model):
        if model == "RNN":
            return rnn_predict(self.text, self.prompt_id)
        elif model == "MLR":
            return mlr_predict(self.text)
        elif model == "BERT":
            return bert_predict(self.text, model_promts[self.prompt_id])  
        elif model == "SVM":
            return svm_predict(self.text)
        elif model == "Todos":
            result = {
                "RNN": rnn_predict(self.text, self.prompt_id),
                "MLR": mlr_predict(self.text),
                "BERT": bert_predict(self.text, self.prompt_id),
                "SVM": svm_predict(self.text)
            }
            return result

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
            
    .img-logo img {
        width: 500px;
        height: 200px;
        object-fit: contain;
    }       
            
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(None, ["Predecir", "Acerca del dataset"],
        icons=['clipboard2-data-fill', 'bar-chart'],
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#000"},
            "icon": {"font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#0e6655"},
            "nav-link-selected": {"background-color": "#16a085"},
        }
    )

if selected == "Predecir":
    st.image("./assets/image.png", use_column_width=True, width=500)

    st.markdown('<h1 class="custom-font">Predicción de contenido y redacción de resumenes</h1>', unsafe_allow_html=True)

    st.markdown('<p class="custom-font"> Selecciona el modelo a utilizar: </p>', unsafe_allow_html=True)
    chosen_model = st.selectbox("", model_options)

    st.markdown('<p class="custom-font"> Selecciona el texto a predecir: </p>', unsafe_allow_html=True)
    input_text = st.text_area(" ")

    if chosen_model == "BERT" or chosen_model == "RNN" or chosen_model == "Todos":
        st.markdown('<p class="custom-font"> Selecciona el prompt a utilizar: </p>', unsafe_allow_html=True)
        promts = list(model_promts.values())
        prompt_selected = st.selectbox("", promts)   

    btn = st.button("Realizar predicción")

    if btn:
        if chosen_model == "MLR":
            pred = mlr_predict(input_text)
            resultados = {
                "Modelo": ["MLR"],
                "Predicción de Contenido": [pred[0]],
                "Predicción de Redacción": [pred[1]],
                "Tiempo de predicción": [pred[2]]
            }
            df_resultados = pd.DataFrame(resultados)
            st.markdown('<h3 class="custom-font">Resultados de la predicción</h3>', unsafe_allow_html=True)
            st.dataframe(df_resultados)

        elif chosen_model == "RNN":
            pred = rnn_predict(input_text, prompt_selected)
            resultados = {
                "Modelo": ["RNN"],
                "Predicción de Contenido": [pred[0]],
                "Predicción de Redacción": [pred[1]],
                "Tiempo de predicción": [pred[2]]
            }
            df_resultados = pd.DataFrame(resultados)
            st.markdown('<h3 class="custom-font">Resultados de la predicción</h3>', unsafe_allow_html=True)
            st.dataframe(df_resultados)

        elif chosen_model == "BERT":
            pred = bert_predict(input_text, prompt_selected)
            resultados = {
                "Modelo": ["BERT"],
                "Predicción de Contenido": [pred[0]],
                "Predicción de Redacción": [pred[1]],
                "Tiempo de predicción": [pred[2]]
            }
            df_resultados = pd.DataFrame(resultados)
            st.markdown('<h3 class="custom-font">Resultados de la predicción</h3>', unsafe_allow_html=True)
            st.dataframe(df_resultados)

        elif chosen_model == "SVM":
            pred = svm_predict(input_text)
            resultados = {
                "Modelo": ["SVM"],
                "Predicción de Contenido": [pred[0]],
                "Predicción de Redacción": [pred[1]],
                "Tiempo de predicción": [pred[2]]
            }
            df_resultados = pd.DataFrame(resultados)
            st.markdown('<h3 class="custom-font">Resultados de la predicción</h3>', unsafe_allow_html=True)
            st.dataframe(df_resultados)
        elif chosen_model == "Todos":
            pred = PrediccionRequest(input_text, prompt_selected).predict("Todos")
            resultados = {
                "Modelo": ["RNN", "MLR", "BERT", "SVM"],
                "Predicción de Contenido": [pred['RNN'][0], pred['MLR'][0], pred['BERT'][0], pred['SVM'][0]],
                "Predicción de Redacción": [pred['RNN'][1], pred['MLR'][1], pred['BERT'][1], pred['SVM'][1]]
            }
            df_resultados = pd.DataFrame(resultados)

            st.markdown('<h3 class="custom-font">Resultados de la predicción</h3>', unsafe_allow_html=True)
            st.dataframe(df_resultados)

            st.markdown('<h3 class="custom-font">Tiempos de predicción</h3>', unsafe_allow_html=True)
            times = {
                "Modelo": ["RNN", "MLR", "BERT", "SVM"],
                "Tiempo de predicción": [pred['RNN'][2], pred['MLR'][2], pred['BERT'][2], pred['SVM'][2]]
            }
            df_times = pd.DataFrame(times)
            fig, ax = plt.subplots()
            sns.barplot(x="Modelo", y="Tiempo de predicción", data=df_times, ax=ax)
            st.pyplot(fig)


elif selected == "Acerca del dataset":
    dataset = pd.read_csv("../data/Finaltrain.csv")
    st.markdown('<h1 class="custom-font">Acerca del dataset</h1>', unsafe_allow_html=True)
    st.code("dataset = pd.read_csv('../data/Finaltrain.csv')")

    st.markdown('<p class="custom-font"> El dataset contiene las siguientes columnas </p>', unsafe_allow_html=True)
    st.write(dataset.head())

    st.markdown('<p class="custom-font"> Estadísticas del dataset </p>', unsafe_allow_html=True)
    st.write(dataset.describe())

    st.markdown('<p class="custom-font"> Distribución de las calificaciones </p>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(dataset["content"], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown('<p class="custom-font"> Distribución de las calificaciones por ensayo </p>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(dataset["wording"], kde=True, ax=ax)
    st.pyplot(fig)
