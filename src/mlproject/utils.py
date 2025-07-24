import pandas as pd
import pymysql
from dotenv import load_dotenv
import os
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import numpy as np
from sklearn.metrics import r2_score

load_dotenv()

username = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
database = os.getenv("DATABASE")

def get_mysql_data():

    try:
        logging.info("Attempting to connect to MySQL database...")
        mydb = pymysql.connect(host=host, user=username, passwd=password, db=database)
        logging.info("Successfully connected to MySQL.")

        df = pd.read_sql_query("SELECT * FROM students", mydb)
        # print(df.head(5))
        return df


    except Exception as e:
        raise CustomException(e)
        logging.info(f"Error connecting to MySQL database: {e}")


def save_object(filename, object):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(object, file)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(X_train, y_train, X_test, y_test, models:dict, params):
    try:

        r2_score_report = {}
        best_params_report = {}
        best_estimator_report = {}

        for model_name, model in models.items():
            parameters_of_model = params.get(model_name, {})

            #cross validation scores
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            gs = GridSearchCV(estimator=model, param_grid=parameters_of_model, cv=kfold, scoring='r2')
            gs.fit(X_train, y_train)
            y_test_pred = gs.predict(X_test)


            #scores
            best_params = gs.best_params_
            best_estimator = gs.best_estimator_
            r2_test_model = r2_score(y_test, y_test_pred)

            r2_score_report[model_name] = r2_test_model
            best_params_report[model_name] = best_params
            best_estimator_report[model_name] = best_estimator

        return r2_score_report, best_params_report, best_estimator_report


    except Exception as e:
        raise CustomException(e)





