import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def mainroot():
    return {"Message":"Hii !"}

@app.get("/model")
def model():
    model = load_model("model.h5")
    y_pred = model.predict([10.0])
    result = int(y_pred[0][0])
    return {"Prediction":result}