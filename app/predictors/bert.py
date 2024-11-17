import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import time

# Definir la clase BERTModel para que coincida con el modelo que entrenaste anteriormente
class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.linear1(pooled_output)
        output = nn.ReLU()(output)
        output = self.linear2(output)
        return output

# Cargar el modelo guardado, mapeándolo al dispositivo correcto
model_path = '../models/bert/bert.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTModel()

# Mapear al dispositivo actual
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Definir el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Cargar el tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict(text, prompt):
    # Tiempo de ejecucion
    start_time = time.time()
    
    # Concatenar el prompt con el texto de entrada
    input_text = f"{prompt} {text}"

    # Tokenizar el texto
    encodings = tokenizer.encode_plus(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Obtener los tensores de input_ids y attention_mask
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    """
    Realiza una predicción de contenido y redacción para una fila de datos.
    :param input_ids: Tensor de IDs de entrada.
    :param attention_mask: Tensor de máscara de atención.
    :return: Resultados de la predicción (content, wording).
    """
    # Mover los tensores al dispositivo
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        content_pred, wording_pred = outputs.cpu().numpy()[0]
    
    end_time = time.time()
    
    # Calcular el tiempo de ejecución
    elapsed_time = end_time - start_time

    return desescalar(content_pred), desescalar(wording_pred), elapsed_time

def desescalar(value):
    """
    Desescala un valor de la predicción. Los datos originales están escalados en un rango de -2 a 5.
    :param value:
    :return:
    """
    return (value + 2) / 7 * 100

if __name__ == '__main__':
    # Prueba
    text = "The third wave was an experimentto see how people reacted to a new one leader government. It gained popularity as people wanted to try new things. The students follow anything that is said and start turning on eachother to gain higher power. They had to stop the experement as too many people got to radical with it blindly following there leader"
    prompt = "Summarize how the Third Wave developed over such a short period of time and why the experiment was ended."

    # Realizar la predicción
    content, wording, tiempoEjecucion = predict(text, prompt)
    print(f"Predicted Content Score: {content:.2f}")
    print(f"Predicted Wording Score: {wording:.2f}")
    print(f"Tiempo de ejecución: {tiempoEjecucion:.2f} segundos")

