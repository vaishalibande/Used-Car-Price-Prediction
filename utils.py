import pandas as pd
import numpy as np
import pickle
import config

def get_used_car_price_prediction(data):
    model_file_path = config.MODEL_FILE_PATH

    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    # Assuming data is a pandas DataFrame
    used_car_price_prediction = model.predict(data)[0]

    return used_car_price_prediction
