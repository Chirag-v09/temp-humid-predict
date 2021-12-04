# -*- coding: utf-8 -*-

# 1. Library imports
import copy

import uvicorn
from fastapi import FastAPI
from payloads import payload
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()

pickle_in = open("HUM_forecast.pkl", "rb")
HUM_forecast = pickle.load(pickle_in)

pickle_in = open("TC_forecast.pkl", "rb")
TC_forecast = pickle.load(pickle_in)

poly_hum = PolynomialFeatures(degree=3)
poly_tc = PolynomialFeatures(degree=4)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'Hello, Recruiter'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict(data: payload):
    data = data.dict()
    obs = int(copy.deepcopy(data['obs']))

    prediction_hum = HUM_forecast.predict(poly_hum.fit_transform([[obs]]))
    prediction_tc = TC_forecast.predict(poly_tc.fit_transform([[obs]]))

    return {
        'prediction_hum': str(prediction_hum[0][0]),
        'prediction_tc': str(prediction_tc[0][0])
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload