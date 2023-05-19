import pandas as pd
import numpy as np
import os
import sys
import yaml
import dill

from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.config import mongo_client


def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database:{database_name} and collection: {collection_name}")
        df = pd.DataFrame(mongo_client[database_name][collection_name].find())
        logging.info(f"Find columns:{df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping columns:_id")
            df = df.drop("_id", axis=1)

        logging.info(f"Rows and Columns in df: {df.shape}")
        return df


    except Exception as e:
        raise InsuranceException(e,sys)
    
def write_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok= True)
        with open(file_path, "w") as file_write:
            yaml.dump(data, file_write)


    except Exception as e:
        raise InsuranceException(e,sys)
    

def convert_columns_float(df:pd.DataFrame, exclude_columns : list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtypes != "O":
                    df[column] = df[column].astype("float")

        return df
    except Exception as e:
        raise InsuranceException(e,sys)


