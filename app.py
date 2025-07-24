from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
from src.mlproject.components.model_predictor import ModelPredictor, ModelPredictorConfig

if __name__ == '__main__':
    try:
        train_data_path, test_data_path, raw_data_path = DataIngestion().get_raw_data()
        train_arr, test_arr = DataTransformation().data_transformation_input(train_data_path, test_data_path)
        r2_scoring = ModelPredictor().get_prediction(train_arr, test_arr)
        print(r2_scoring)


    except Exception as e:
        raise CustomException(e)