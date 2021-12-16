from fastapi import FastAPI, encoders
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
import os
import pickle
import pandas as pd

# Instantiate the app.
app = FastAPI()

# Heroku access to DVC data
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class Input(BaseModel):
    age : int
    workclass : str
    fnlgt : int
    education : str
    education_num : int
    marital_status : str
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int
    capital_loss : int
    hours_per_week : int
    native_country : str

class Output(BaseModel):
    prediction:str


gb_model = pickle.load(open("./model/gbclassifier.pkl", "rb"))
rf_model = pickle.load(open("./model/randomforest.pkl", "rb"))
encoder = pickle.load(open("./model/encoder.pkl", "rb"))
lb = pickle.load(open("./model/lb.pkl", "rb"))

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    msg = "Hi, this is the API for model inference on census data, created by Samuel Amico"
    return msg

@app.post("/items/")
async def inference(data: Input):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # load predict_data
    request_dict = data.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])
    
    X, _, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                training=False,
                encoder=encoder,
                lb=lb)

    prediction = inference(X)

    if prediction[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"prediction": prediction}