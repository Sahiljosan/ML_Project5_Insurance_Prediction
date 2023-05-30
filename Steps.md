# ML_Project5_Insurance_Prediction
- [Insert Data into MongoDB](#insert-data-into-mongodb)
- [Get data from MongoDB](#get-data-from-mongodb)
- [Day 3](#day-3)
- [Day 4 Data_ingestion and Data_Validation](#day-4)
- [Day 5 Data_transformation](#day-5-data_transformation)
- [Day 6 Model_Training and Model_evaluation](#day-6-model_training-and-model_validation)
- [Day 7 Model Evaluation Part 2 and model pusher](#day-7-model-evaluation-part-2-and-model-pusher)
- [Day 8 Batch Prediction, Training Pipeline, and Web Application](#day-8-batch-prediction-training-pipeline-and-web-application)
- [Streamlit Web app](#create-streamlit-web-app)

## Insert Data into MongoDB
Create file data_dump.py and write below code to dump data in to mongo.DB
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
In utils.py write below code to fetch data from mongoDB
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
<br>
Write code in setup.py file
<br>
Define MONGO_CLIENT in Insurance_prediction/config.py file

## Day 3
After logger and Exception handling , we will fetch data from mongoDB <br>
`step1` go to utils and write code <br>
`Step2` Create one file .env and assign MONGO_DB_URL. <br>
```
MONGO_DB_URL = "mongodb+srv://sahil_josan:samongodbhil5@cluster0.sptya9h.mongodb.net/?retryWrites=true&w=majority"
``` 
<br>
go to Insurance_Predcition/__init__.py file and write code 

```
from dotenv import load_dotenv
# we have to install dotenv library in cmd
print(f"Loading env variable from .env")
load_dotenv()
```

`step3` go to Insurance_Prediction.config.py file <br>

In order to read data from mongoDB, we will define our client in mongoDB from where we are going to fetch the data.<br>
`step4` Write code in main.py <br>

```
if __name__ == "__main__":
    try:
        get_collection_as_dataframe(database_name= "INSURANCE", collection_name = "INSURANCE_PROJECT")

    except Exception as e:
        print(e)
```

`step5` install all libraries using pip install -r requirements.txt (copy paste) <br> 
`step6` Run python main.py file <br>
After this all data will be ingested from mongoDB server <br>
`Step7` Create 2 folders in Insurance_Prediction/entity : artifact_enity.py and config_entity.py <br>
Now we will write code in config.py file <br>
`step7` Now we will make one folder of **data_set** name, where we will split data into train data and split data <br>

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

`step4` Go to utils.py and define our convert_columns_float, write code from line 28 - 37<br>
`step5` Again go to data_validation.py and write code from 114 - 128<br>
`step6` Go to utils.py and define function write_yaml_file <br>
`step7` in data_validation.py write code from 129 - 136<br>
`step8` in main.py write code from 38 - 44
<br>
## Day 5 Data_Transformation
Here we will
- Handle Missing Values
- Handle Outliers
- Handle Inbalance Data 
- Convert categorical data into numerical data
- Based on this data we will train our model (Model_Building)

`step1` Go to Insurance_Prediction/entity/config_entity.py and write code from line 53 - 60<br>
`step2` In artifacts_entity.py make class DataTransformationArtifacts<br>
`step3` go to data_transformation.py and write code from DataTransformation class and get_data_transformer_object function and initiate_data_transformation function<br>
`step4` go to utils.py and define save_object function , load_object function and save_numpy_array_data function<br>
`step5` go to data_transformation.py and write code from 104 to 116 data_transformation_artifact<br>
`step6` go to main.py and write code for data_transformation <br>
After This we get error like `expected str, bytes or os.PathLike object, not TrainingPipelineConfig`, so to remove this error <br>
- go to utils and define save_numpy_array_data function
- go to data_transformation.py and write code from 107 - 118 
- go to config_entity.py and write code in line 63

## Day 6 Model_Training and Model_Validation
**ModelTrainer**
`step1` go to entity/config_entity.py and define ModelTrainingConfig class <br>
`step2` go to entity/artifact_entity.py and define ModelTrainerArtifact class <br>
`step3` go to model_trainer.py and write code for ModelTrainer class and define train_model and initiate_model_trainer <br>
`step4` go to utils.py file and write code to **Load Data in model Trainer file** <br>
`step5` go to model_trainer.py and write code function **initiate_model_trainer** <br>
`step6` go to main.py and write code for model trainer <br>

Mow we have created 1st pipeline where we have data_ingestion, data_transformation,data_validation and model_trainer and we can deply the model. <br>
Right now we are saving all things in artifact folder <br>
But after model deployment, next time when we get the data ,we will again run the pipeline so we have to write code so that when the new data come it will synchronize with the older one <br>
In **predictor.py** we will create new class **ModelResolver**, so when we run new pipeline with new data then this class will create a new folder <br>
After Creating folder this class will save the new model <br>
we cannot save the new model with the previous one .. because if we do so, then how can we compare new model with the previous one <br>
So we will compare our new model with the older one and after comparing if we get the better accuracy then old model.<br>
then we will accept this new model else reject this new model<br>

**Model Validation**
`step1` go to Insurance_Prediction/predictor.py and define class ModelTrainer 
and define all functions <br>
`step2` go to artifact_entity.py and define class ModelEvaluationArtifact <br>
`step3` go to config_entity.py file and define **ModelEvaluationConfig** class <br>
`step4` create model_evaluation.py file and define class ModelEvaluation in it , write code for constructor and define intiate_model_evaluation function<br>
`step5` go to main.py and write code for Model Evaluation
`step6` Now our folder is created "saved_models" but it will be empty


## Day 7 Model Evaluation Part 2 and Model Pusher
Whenever we get the new data, the new model will be trained using the new data and we get the new accuracy . <br>
if the new accuracy is better then the previous accuracy then the new model will be deployed and if new accuracy is less then the previous accuracy then previous model will be deployed <br>

`step1` go to model_evaluation.py and write code from **Find location of previous model** to **return model_eval_artifact** <br>
`step2` create file model_pusher.py <br>
`step3` go to config_entity.py and write code for class **ModelPusherConfig** <br>
`step4` go to artifact_entity.py and write code for class **ModelPusherArtifact** <br>
`step5` go to Insurance_Prediction/components/model_pusher.py  and write code for class **ModelPusher** <br> 
Create constructor in class and define **initiate_model_pusher** function
`step6` go to main.py file and write code for model pusher

## Day 8 Batch Prediction, Training Pipeline, and Web Application
`step1` Go to Insurance_Prediction/pipeline and create file Batch Prediction <br>
and define **start_batch_prediction**
`step2` Create demo.py file to check the batch_prediction.py <br>
After running python demo.py , we will get a prediction folder where the output of batch prediction.py is available <br>
`step3` 
all functions that we have defined in component folder, we will define it in training_pipeline.py <br>
Because when the user will come, and run the training_pipeline.py file then everything data_ingestion, data_validation, data_transformation , model_trainer, model_evaluation will be done <br>
and then the user will do batch_prediction <br>
`step4` Now in Insurance_Prediction/pipeline folder create file traning_pipeline.py and define function <br>
define functions for data_ingestion, data_validation, data_transformation , model_trainer, model_evaluation, model_pusher

## Create Streamlit web app
`step1` create app.py file <br>
Write code 
```
import streamlit as st
st.title("Insurance Premium Prediction")
```
<br>
To check the streamlit is working or not , write 
```
streamlit run app.py
```
<bt>
- To create web application we need pickle file, which is available in Insurance_Prediction/saved_model.py
- In industry we use model registry, which is available in any cloud platform and from there we are fetching the pickle file
- Bt now we are working in local system, we will copy model.pkl , target_encoder.pkl and transformer.pkl file from saved_models and paste it in main ouput folder, near main.py and setup.py <br>

`step2` Now we will load model.pkl , target_encoder.pkl and transformer.pkl one by one in app.py
















