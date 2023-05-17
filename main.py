from Insurance_Prediction.logger import logging
from Insurance_Prediction.exception import InsuranceException
import os,sys

from Insurance_Prediction.utils import get_collection_as_dataframe
from Insurance_Prediction.entity.config_entity import DataIngestionConfig
from Insurance_Prediction.entity import config_entity


# def test_logger_and_exception():
    # try:
    #     logging.info("Starting the test logger and exception")
    #     result = 3/0
    #     print(result)
    #     logging.info("Ending point of the logger and exception class")
    # except Exception as e:
    #     logging.debug(str(e))
    #     raise InsuranceException(e, sys)
    

if __name__ == "__main__":
    try:
        # start_training_pipeline()
        # test_logger_and_exception()
        # get_collection_as_dataframe(database_name = "INSURANCE", collection_name = "INSURANCE_PROJECT")
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        print(data_ingestion_config.to_dict())
    except Exception as e:
        print(e)
        