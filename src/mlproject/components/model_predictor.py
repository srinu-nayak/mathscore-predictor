from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import os
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.mlproject.utils import evaluate_models
from src.mlproject.utils import save_object



@dataclass
class ModelPredictorConfig:
    model_path: str = os.path.join('artifacts','model.pkl')

class ModelPredictor:
    def __init__(self):
        self.model_config= ModelPredictorConfig()

    def get_prediction(self, train_arr, test_arr):
        try:
            X_train = train_arr[:, : -1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, : -1]
            y_test = test_arr[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },"AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            r2_score_report, best_params_report, best_estimator_report = evaluate_models (
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_score = max(sorted(r2_score_report.values()))
            logging.info(f"Best model: {best_model_score}")

            best_model_name = list(r2_score_report.keys())[list(r2_score_report.values()).index(best_model_score)]
            best_model = best_estimator_report[best_model_name]
            logging.info(f"Best model: {best_model}")

            best_hyperparameters = best_params_report[best_model_name]
            logging.info(f"Best hyperparameters: {best_hyperparameters}")

            print(best_model_score)
            print(best_model)
            print(best_hyperparameters)


            save_object(
                self.model_config.model_path,
                best_model,
            )
            logging.info(f'saving model to artifacts using pickle')

            y_hat = best_model.predict(X_test)
            return r2_score(y_test, y_hat)


        except Exception as e:
            raise CustomException(e)

