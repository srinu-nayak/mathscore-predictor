from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig


if __name__ == '__main__':
    try:
        data_ingestion = DataIngestion().get_raw_data()

    except Exception as e:
        raise CustomException(e)