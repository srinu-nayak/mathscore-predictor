import os.path
from src.mlproject.utils import get_mysql_data
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_raw_data(self):
        try:
            df = get_mysql_data()
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved to {}'.format(os.path.abspath(self.ingestion_config.raw_data_path)))


            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Train data saved to {os.path.abspath(self.ingestion_config.train_data_path)}')

            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f'Test data saved to {os.path.abspath(self.ingestion_config.test_data_path)}')


            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e)






