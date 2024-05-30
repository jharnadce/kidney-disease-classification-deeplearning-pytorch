from cnnClassifierKidneyDisease import logger
from cnnClassifierKidneyDisease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
obj = DataIngestionTrainingPipeline()
obj.main()
logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=========x")