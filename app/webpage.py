# Importaciones necesarias
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model

nltk.download('stopwords')
print(stopwords.words('english'))
stops = set(stopwords.words('english'))
print(stops)

# Configuración de los modelos disponibles
model_options = {
    "MLR": {
        "content_model": "../models/mlr/modelo_content.joblib",
        "wording_model": "../models/mlr/modelo_wording.joblib"
    },
    "SVR": {
        "content_model": "../models/svr/modelo_content.pkl",
        "wording_model": "../models/svr/modelo_wording.pkl"
    },
    "RNN": {
        "content_model": "../models/rnn",
    },
}

model_promts = {
    "39c16e": "Summarize at least 3 elements of an ideal tragedy, as described by Aristotle.",
    "3b9047": "In complete sentences, summarize the structure of the ancient Egyptian system of government. How were different social classes involved in this government? Cite evidence from the text.",
    "814d6b": "Summarize how the Third Wave developed over such a short period of time and why the experiment was ended.",
    "ebad26": "Summarize the various ways the factory would use or cover up spoiled meat. Cite evidence in your answer.",
}

model_prompt_titles = ["The Third Wave", "Excerpt from The Jungle", "Egyptian Social Structure", "On Tragedy"]

model_prompt_text = [ "Background \n The Third Wave experiment", "With one member trimming beef", "Egyptian society was structured", "Chapter 13 \n As the sequel"]

# Clase para manejar la predicción con dos modelos
class PrediccionRequest:
    def __init__(self, content_model_path, wording_model_path, rnn_model, text: str, prompt_id: str):
        self.content_model = self.load_model(content_model_path)
        self.wording_model = self.load_model(wording_model_path)
        self.rnn_model = load_model(model_options["RNN"])
        self.text = text
        self.prompt_id = prompt_id
        self.features = self.preprocess_text()

    def load_model(self, model_path):
        # Cargar el modelo dependiendo de su extensión
        if model_path.endswith('.joblib'):
            return joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Formato de archivo no soportado. Usa '.joblib' o '.pkl'.")

    # student_id,prompt_id,text,content,wording,prompt_question,prompt_title,prompt_text,text_length,word_count,number_count,punctuation_count,stopword_count
    def preprocess_text(self):
        # Procesamiento de texto para extraer características
        if chosen_model == "RNN":
            return self.text
        else:
            features = {
                "student_id": int("000e8c3c7ddb", 16),
                "prompt_id": int(self.prompt_id, 16),
                "text_length": len(self.text),
                "word_count": len(self.text.split()),
                "number_count": len([word for word in self.text.split() if word.isnumeric()]),
                "punctuation_count": len([char for char in self.text if char in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '"', "'"]]),
                "stopword_count": len([word for word in self.text.split() if word in stopwords.words('english')])
            }
            print(features)
            return np.array(list(features.values()))

    def predict(self):
        content_prediction = self.content_model.predict(self.features.reshape(1, -1))
        wording_prediction = self.wording_model.predict(self.features.reshape(1, -1))
        return content_prediction[0], wording_prediction[0]  # Retorna ambas predicciones

chosen_model = st.sidebar.selectbox("Selecciona el modelo para la predicción:", list(model_options.keys()))
if chosen_model == "RNN":
    rnn_model_path = model_options[chosen_model]
    # dropdown de los prompts
else:
    content_model_path = model_options[chosen_model]["content_model"]
    wording_model_path = model_options[chosen_model]["wording_model"]
    prompt_val = option_menu("Selecciona el prompt a usar:", list(model_promts.values()))

    for key, value in model_promts.items():
        if value == prompt_val:
            prompt_id = key

    st.write(f"Prompt seleccionado: {prompt_id}")

    # dropdown de los prompts_title
    prompt_title_val = option_menu("Selecciona el prompt_title a usar:", model_prompt_titles)

    # dropdown de los prompts_text
    prompt_text_val = option_menu("Selecciona el prompt_text a usar:", model_prompt_text)

st.sidebar.success(f"Modelos de '{chosen_model}' seleccionados para 'content' y 'wording'.")

st.title("Predicción de contenido y calidad de textos, usando modelos de: " + chosen_model)


input_text = st.text_area("Introduce el texto para la predicción:")
if st.button("Realizar Predicción"):
    # Crear la instancia de predicción con ambos modelos
    if chosen_model == "RNN":
        prediccion_request = PrediccionRequest( None, None,rnn_model_path, input_text, prompt_id)
    else:
        prediccion_request = PrediccionRequest(content_model_path, wording_model_path, input_text, prompt_id)

    content_pred, wording_pred = prediccion_request.predict()
    
    # Mostrar los resultados
    st.write(f"Predicción de 'content': {content_pred}")
    st.write(f"Predicción de 'wording': {wording_pred}")