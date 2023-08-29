import sys
import os

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensamble import(
    AdaBoostRegressor,
    GardientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegresson
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighboursRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging 



from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def intiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting test and train input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models=(
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Booster" : GardientBoostingRegressor(),
                "Linear Regression" : LinearRegresson(),
                "K - Neibours Classifier" : KNeighboursRegressor(),
                "XGBClassifier" : XGBClassifier(),
                "Cat Boosting Classifier": CatBoostRegressor(verbose=False),
                "Ada Boost Classifier" : AdaBoostRegressor(),
            )

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,
            x_test=x_test,y_test=y_test,models=models)

            #to get best model score from dict
            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info(f"Best found model on Train and Test datasets ")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square

        except Exception e:
            raise CustomException(e,sys)


