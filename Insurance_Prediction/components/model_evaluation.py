import os,sys
import pandas as pd
import numpy as np

from Insurance_Prediction.entity import config_entity,artifact_entity
from Insurance_Prediction import utils
from Insurance_Prediction.utils import load_object
from Insurance_Prediction.config import TARGET_COLUMN
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.predictor import ModelResolver
from Insurance_Prediction.logger import logging
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



class ModelEvaluation:
    def __init__(self,
                 model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact : artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact : artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact : artifact_entity.ModelTrainerArtifact):
        
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise InsuranceException(e,sys)
        

    def intiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("if saved model folder has model the we will compare"
                         "Which model is best trained or the model from saved model folder")

            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True, improved_accuracy = None)
                logging.info(f"Model Evaluation Atifact: {model_eval_artifact}")

                return model_eval_artifact
            
            # Find location of previous model
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            # All this we are defining for previous model
            transformer = load_object(file_path = transformer_path)
            model = load_object(file_path = model_path)
            target_encoder = load_object(file_path = target_encoder_path)

            # defining for New model
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]

            y_true = target_df

            input_features_name = list(transformer.feature_names_in)
            for i in input_features_name:
                if test_df[i].dtypes == "O":
                    test_df[i] = target_encoder.fit_transform(test_df[i])

            input_arr = transformer.transform(test_df[input_features_name])
            y_pred = model.predict(input_arr)

            # comparision b/w new model and old model

            previous_model_score = r2_score(y_true=y_true, y_pred= y_pred)

            # Accuracy of current model

            input_features_name = list(current_transformer.feature_names_in)
            input_arr = current_transformer.transform(test_df[input_features_name])
            y_pred = current_model.predict(input_arr)
            y_true = target_df

            current_model_score = r2_score(y_true= y_true, y_pred= y_pred)

            # Final comparision between accurcy of both models
            if current_model_score < previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current model is not better then previous model")
            

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted= True,
                                                                          improved_accuracy=current_model_score - previous_model_score)
            
            return model_eval_artifact


        except Exception as e:
            raise InsuranceException(e,sys)

# After comparing the model we have to store the model some cloud
# cloud (AWS -> s3 bucket)
# Database -> Model pusher
        
