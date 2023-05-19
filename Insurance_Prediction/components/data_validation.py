import os,sys
import pandas as pd
import numpy as np
from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
from Insurance_Prediction.entity import config_entity,artifact_entity
from typing import Optional
from Insurance_Prediction.config import TARGET_COLUMN

from Insurance_Prediction import utils

from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self, data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        
        try:
            logging.info(f"****************Data Validation*****************")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e:
            raise InsuranceException(e,sys)




    def drop_missing_values_columns(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            drop_columns_names = null_report[null_report > threshold].index

            self.validation_error[report_key_name] = list(drop_columns_names)
            df.drop(list(drop_columns_names), axis = 1, inplace= True)

            if len(df.columns) == 0:
                return None
            return df

        except Exception as e:
            raise InsuranceException(e,sys)




    def is_required_columns_exists(self,base_df:pd.DataFrame, current_df : pd.DataFrame,report_key_name:str)->bool:
        try:
            base_columns = base_df
            current_columns = current_df

            missing_columns = []
            for base_columns in base_columns:
                if base_columns not in current_columns:
                    logging.info(f"column:[{base} is not available ]")
                    missing_columns.append(base_columns)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True

        except Exception as e:
            raise InsuranceException(e,sys)





    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name : str): # base_df:pd.DataFrame means pd.Dataframe is defined as base_df
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue > 0.05:
                    # Null hypothesis accept
                    drift_report[base_column] = {
                        "pvalues" : float(same_distribution.pvalue),
                        "same_distribution" : True
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues" : float(same_distribution.pvalue),
                        "same_distribution" : False
                    }
                    # different distribution

            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise InsuranceException(e,sys)

    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact: # initiate_data_validation will comes from artifact_entity.DataValidationArtifact
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace= True)
            #base_df has na as null
            base_df = self.drop_missing_values_columns(df = base_df,report_key_name= "Missing_values_within_base_dataset")
            

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df = train_df,report_key_name= "Missing_values_within_train_dataset")
            test_df = self.drop_missing_values_columns(df = test_df,report_key_name= "Missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df = base_df, exclude_columns= exclude_columns)
            train_df = utils.convert_columns_float(df = train_df, exclude_columns= exclude_columns)
            test_df = utils.convert_columns_float(df = test_df, exclude_columns= exclude_columns)

            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df, report_key_name="Missing_columns_within_train_dataset")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="Missing_columns_within_test_dataset")

            if train_df_columns_status:
                self.data_drift(base_df=base_df,current_df=train_df, report_key_name="data_drift_within_train_dataset")
                
            if test_df_columns_status:
                self.data_drift(base_df= base_df, current_df= test_df, report_key_name="data_drift_within_test_dataset")
            
            # write your report
            utils.write_yaml_file(file_path= self.data_validation_config.report_file_path,
                                  data = self.validation_error)
            
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path = self.data_validation_config.report_file_path)
            
            return data_validation_artifact


        except Exception as e:
            raise InsuranceException(e,sys)





