from sklearn import preprocessing
import numpy as np
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.mlproject.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessing_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_config(self, input_features_train_df):
        numerical_columns = input_features_train_df.select_dtypes(include='number').columns
        categorical_columns = input_features_train_df.select_dtypes(exclude='number').columns

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())

        ])

        ordinal_pipeline = Pipeline(steps=[
            ('ordinal', OrdinalEncoder()),
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        onehot_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore', sparse_output=False)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        transformer = ColumnTransformer(
            transformers=[
                ('numerical', numerical_pipeline, [5, 6]),
                ('ordinal', ordinal_pipeline, [2]),
                ('onehot', onehot_pipeline, [0, 1, 3, 4]),

            ],
            remainder='passthrough',
        )

        return transformer


    def data_transformation_input(self, train_path, test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('dividing independent and dependent features into train and test sets')
            input_features_train_df = train_df.drop(columns=['math_score'])
            target_features_train_df = train_df['math_score']
            #converting series to array
            target_features_train_df = target_features_train_df.values.reshape(-1, 1)

            input_features_test_df = test_df.drop(columns=['math_score'])
            # converting series to array
            target_features_test_df = test_df['math_score']
            target_features_test_df = target_features_test_df.values.reshape(-1,1)


            logging.info('preprocessing data')

            preprocessing_object = self.get_data_transformation_config(input_features_train_df)

            input_feature_train_arr = preprocessing_object.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_features_test_df)

            train_arr = np.concatenate((input_feature_train_arr, target_features_train_df), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_features_test_df), axis=1)


            logging.info('saving data to artifacts using pickle')
            save_object(
                self.data_transformation_config.preprocessing_object_file_path,
                preprocessing_object
            )



            logging.info('preprocessing data sending to app.py')
            return train_arr, test_arr


        except Exception as e:
            raise CustomException(e)

