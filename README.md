# ML_Project5_Insurance_Prediction
- [Insert Data into MongoDB](#insert-data-into-mongodb)
- [Get data from MongoDB](#get-data-from-mongodb)
- [Day 3](#day-3)
- [Day 4](#day-4)
- []

## Insert Data into MongoDB
```
import pymongo 
import pandas as pd
import json

client = pymongo.MongoClient("mongodb+srv://sahil_josan:samongodbhil5@cluster0.sptya9h.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH = (r"G:\Udemy\DATA SCIENCE ineuron\VS Code\ML_Project5_Insurance_Prediction\insurance.csv")

DATABASE_NAME = "INSURANCE"
														
COLLECTION_NAME = "INSURANCE_PROJECT"					


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    df.reset_index(drop = True, inplace = True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

```

## Get data from mongoDB
```
import pandas as pd
import numpy as np
import os
import sys

from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.config import mongo_client


def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
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


```

## Day 3
After logger and Exception handling , we will fetch data from mongoDB <br>
`step1` go to utils and write code
`Step2` Create one file .env and store mongoDB client.
`step3` go to Insurance_Prediction.config.py file <br>
In order to read data from mongoDB, we will define our client in mongoDB from where we are going to fetch the data.
`step4` Write code in main.py 
`step5` install all libraries using pip install -r requirements.txt
`step6` Run python main.py file <br>
After this all data will be ingested from mongoDB server
`Step7` Create 2 folders in Insurance_Prediction/entity : artifact_enity.py and config_entity.py <br>
Now we will write code in config.py file
`step7` Now we will make one folder of **data_set** name, where we will split data into train data and split data

## Day 4
`step1` Create file data_ingestion.py in Insurance_Prediction/components folder <br>
Write code in data_ingestion.py file
`step2` change code in the main.py file <br>
After running python main.py we will get artifact folder, which includes train and test dataset and insurance.csv dataset<br>
`step3` Create folder Data Validation in Insurance_Prediction/components <br>
In data validation we :
- Check Data Type
- Find the unwanted data 
- Data Cleaning
<br>
Open Insurance_Prediction\entity\config_entity.py file and write code for data_validation<br>
Now go to Insurance_Prediction\entity\artifact_entity.py and define data_validation artifact <br>
Now again go data_validation.py file and write code upto line 111 <br>

`step4` Go to utils.py and define our convert_columns_float, write code from line 28 - 37
`step5` Again go to data_validation.py and write code from 114 - 128
`step6` Go to utils.py and define function write_yaml_file 
`step7` in data_validation.py write code from 129 - 136
`step8` in main.py write code from 38 - 44
