import numpy as np
import pandas as pd

import tensorflow as tf
import pickle


def load_models():
    model = tf.keras.models.load_model("../models/rnn.keras")
    with open("../models/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict(text: str, prompt: str) -> tuple[float, float]:
    """
    Realiza una predicción de contenido y redacción para una fila de datos.
    :param text: Texto del estudiante.
    :param prompt: Pregunta asociada al texto.
    :return: Resultados de la predicción (content, wording).
    """
    MODEL, TOKENIZER = load_models()

    row_df = pd.DataFrame([{'text': text, 'prompt_text': prompt}])

    text_data = row_df['text'] + ' ' + row_df['prompt_text']
    sequences = TOKENIZER.texts_to_sequences(text_data)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)

    prediction = MODEL.predict(padded_sequence, verbose=0)

    content_pred, wording_pred = prediction

    print(content_pred, wording_pred)

    content_pred = desescalar(content_pred[0][0])
    wording_pred = desescalar(wording_pred[0][0])

    return content_pred, wording_pred


def desescalar(value):
    """
    Desescala un valor de la predicción. Los datos originales están escalados en un rango de -2 a 5.
    :param value:
    :return:
    """
    return (value + 2) / 7 * 100


if __name__ == '__main__':
    text = "The third wave only started as an experiment within the class but it slowly spread through kids partipating outside of class. Some kids we're even reporting back to Mr. Jones if another student didn't abide by the rules. As more and more kids joined the \"movement\" Mr. Jones realized it was slipping out of control so he terminated the movement."
    prompt = "Summarize how the Third Wave developed over such a short period of time and why the experiment was ended."
    content_pred, wording_pred = predict(text, prompt)

    print(f"Content: {content_pred:.2f}%")
    print(f"Wording: {wording_pred:.2f}%")
