from cnnClassifierKidneyDisease import logger
from cnnClassifierKidneyDisease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifierKidneyDisease.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
STAGE_NAME = "Data Ingestion stage"

logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
obj = DataIngestionTrainingPipeline()
obj.main()
logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=========x")

STAGE_NAME = "Prepare Base Model"

logger.info(f"**********************")
logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
obj = PrepareBaseModelTrainingPipeline()
obj.main()
logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=======x")