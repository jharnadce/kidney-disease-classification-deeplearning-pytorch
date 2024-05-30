import os
import zipfile
from cnnClassifierKidneyDisease import logger
import gdown
from cnnClassifierKidneyDisease.entity.config_entity import DataIngestionConfig

class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the url
        """
        try:
            print(self.config)
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading from {dataset_url} into {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix_url = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix_url+file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            raise e
    
    def extract_zip_file(self):
        """
        zip file path is a str
        extracts zip file and saves to data directory
        function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
