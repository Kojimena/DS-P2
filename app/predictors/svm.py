import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# Definir la lista de palabras vacías
stops = set(stopwords.words('english'))

# Rutas de los modelos
CONTENT_MODEL_PATH = 'models/svm/svr_content.joblib'
WORDING_MODEL_PATH = 'models/svm/svr_wording.joblib'
SCALER_PATH = 'models/svm/scaler.joblib'

# Cargar los modelos
try:
    svr_content = joblib.load(CONTENT_MODEL_PATH)
    svr_wording = joblib.load(WORDING_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modelos cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error al cargar los modelos: {e}")
    exit(1)


def predict(text: str) -> tuple[float, float]: 
    """
    Realiza una predicción de contenido y redacción para una fila de datos.
    :param text: Texto del estudiante.
    :return: Resultados de la predicción (content, wording).
    """
    # Extraer características
    features = extract_features(text)
    feature_array = pd.DataFrame([features])
    feature_scaled = scaler.transform(feature_array)
    
    # Predecir
    content_pred = svr_content.predict(feature_scaled)
    wording_pred = svr_wording.predict(feature_scaled)

    return desescalar(content_pred[0]), desescalar(wording_pred[0])
    
    
def extract_features(text: str) -> dict:
    """
    Extrae características del texto.
    :param text: Texto del estudiante.
    :return: Características extraídas.
    """
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "number_count": len([word for word in text.split() if word.isnumeric()]),
        "punctuation_count": len([char for char in text if char in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '"', "'"]]),
        "stopword_count": len([word for word in text.split() if word in stops])
    }

def desescalar(value):
    """
    Desescala un valor de la predicción. Los datos originales están escalados en un rango de -2 a 5.
    :param value:
    :return:
    """
    return (value + 2) / 7 * 100

if __name__ == '__main__':
    text = "The Third Wave developed  rapidly because the students genuinly believed that it was the best course of action. Their grades, acomplishments, and classparticipation/ behavior had improved dramatically since the experiment began. There did not seem to be any consiquenses in the students eyes. They became extremely engaged in all the Third Wave activites both inside and outside tha classroom. The experiment ended because the students were so patriotic about the \"movement\". The history class of thirty rapidly grew to 200 in three days.  That means 170 students joined a school \"movement\" in two days. Thats 85 people per day! On the fifth and final day all the students had completley believed that the \"Third Wave\" was a movement that would expell democracy. They believed a candidate from the \"movement\" would anounce its existance on television after five days of its success. The creater, Ron Jones, believed it had gone too far and for everyone's safety he shut it down.  If he hadn't the fake organization would have grown into something out of his controll. The Third Wave only lasted for a week. It could have spiralled into the American version of the Nazi Party, which is the opposite of what America stands for."
    print (predict(text))