import os,sys
import pandas as pd
import numpy as np
from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.predictor import ModelResolver
from Insurance_Prediction.utils import load_object
from Insurance_Prediction.entity import config_entity,artifact_entity
from typing import Optional
from datetime import datetime


PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok= True)
        model_resolver = ModelResolver(model_registry= "saved_models")

        # Data Loading
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN}, inplace= True)

        # Data Validation
        transformer = load_object(file_path = model_resolver.get_latest_transformer_path())

        target_encoder = load_object(file_path= model_resolver.get_latest_target_encoder_path())

        input_features_names = list(transformer.feature_names_in_)
        # inport categorical data in our input_features
        for i in input_features_names:
            if df[i].dtypes == "object":
                df[i] = target_encoder.fit_transform(df[i])
        
        input_arr = transformer.transform(df[input_features_names])

        model = load_object(file_path = model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        df['prediction'] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv") # replace data with .csv format and the file that create will be with current date and time

        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)

        df.to_csv(prediction_file_path, index = False, header = True) # load csv with index = False and header = True

        return prediction_file_path

    except Exception as e:
        raise InsuranceException(e,sys)
    

    

