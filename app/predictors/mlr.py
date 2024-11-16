import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pickle
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))
stops = set(stopwords.words('english'))
print(stops)

content_model = "../../models/mlr/modelo_content.joblib"
wording_model = "../../models/mlr/modelo_wording.joblib"
scaler = "../../models/mlr/scaler.joblib"
poly = "../../models/mlr/poly_features.joblib"

feature_scaler = joblib.load(scaler)
poly_features = joblib.load(poly)

def predict(text, promt_id) -> tuple[float, float]:
    # Load text, student_id, prompt_id    content = joblib.load(content_model)

    # Extract features
    features = extract_features(text)
    features["prompt_id"] = promt_id
    # features["student_id"] = student_id
    
    feature_array = np.array(list(features.values())).reshape(1, -1)
    feature_scaled = feature_scaler.transform(feature_array)
    feature_poly = poly_features.transform(feature_scaled)

    # Predict
    content = joblib.load(content_model)
    wording = joblib.load(wording_model)

    content_pred = content.predict(feature_poly)
    wording_pred = wording.predict(feature_poly)

    return content_pred[0], wording_pred[0]

def extract_features(text: str) -> dict:
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "number_count": len([word for word in text.split() if word.isnumeric()]),
        "punctuation_count": len([char for char in text if char in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '"', "'"]]),
        "stopword_count": len([word for word in text.split() if word in stopwords.words('english')])
    }


if __name__ == '__main__':

    # Test
    text = "The Third Wave developed  rapidly because the students genuinly believed that it was the best course of action. Their grades, acomplishments, and classparticipation/ behavior had improved dramatically since the experiment began. There did not seem to be any consiquenses in the students eyes. They became extremely engaged in all the Third Wave activites both inside and outside tha classroom. The experiment ended because the students were so patriotic about the \"movement\". The history class of thirty rapidly grew to 200 in three days.  That means 170 students joined a school \"movement\" in two days. Thats 85 people per day! On the fifth and final day all the students had completley believed that the \"Third Wave\" was a movement that would expell democracy. They believed a candidate from the \"movement\" would anounce its existance on television after five days of its success. The creater, Ron Jones, believed it had gone too far and for everyone's safety he shut it down.  If he hadn't the fake organization would have grown into something out of his controll. The Third Wave only lasted for a week. It could have spiralled into the American version of the Nazi Party, which is the opposite of what America stands for."
    prompt_id = int("814d6b", 16)

    print(predict(text, prompt_id))