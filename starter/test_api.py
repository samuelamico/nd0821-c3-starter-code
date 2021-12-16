from fastapi.testclient import TestClient
from main import app
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

client = TestClient(app)

# A function to test the get
def test_get():
    """
    Test reading the root msg
    """
    try:
        response = client.get("/")
    except TypeError:
        logger.error("Root path not found")
    
    validate_content = response.content.decode('utf-8').strip('"')
    assert response.status_code == 200
    assert validate_content == "Hi, this is the API for model inference on census data, created by Samuel Amico"


def test_post_less_50k():
    """
    Test the predict output for salary >=50k.
    """
    input_dict = {
                    "age": 49,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary <= 50k"


def test_post_greater_50k():
    input_dict = {
                    "age": 31,
                    "workclass": "Private",
                    "fnlgt": 45781,
                    "education": "Masters",
                    "education_num": 14,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 1020,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary > 50k"