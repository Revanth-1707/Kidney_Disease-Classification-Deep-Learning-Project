import os
import zipfile
import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
        
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Download dataset only if zip file doesn't exist
        """

        try:

            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(
                "artifacts/data_ingestion",
                exist_ok=True
            )

            if os.path.exists(zip_download_dir):

                logger.info(
                    f"File already exists: {zip_download_dir}"
                )

                return

            logger.info(
                f"Downloading file from [{dataset_url}] into file {zip_download_dir}"
            )

            file_id = dataset_url.split("/")[-2]

            prefix = (
                "https://drive.google.com/uc?export=download&id="
            )

            gdown.download(
                prefix + file_id,
                zip_download_dir
            )

            logger.info(
                f"Downloaded data from {dataset_url}"
            )

        except Exception as e:
            raise e

    def extract_zip_file(self):

        unzip_file_path = self.config.unzip_dir

        dataset_folder = os.path.join(
            unzip_file_path,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        if os.path.exists(dataset_folder):

            logger.info(
                "Dataset already extracted. Skipping extraction."
            )

            return

        os.makedirs(
            unzip_file_path,
            exist_ok=True
        )

        with zipfile.ZipFile(
            self.config.local_data_file,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                unzip_file_path
            )

        logger.info(
            "Dataset extracted successfully."
        )
