import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from a file path using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_regression_model(y_true, y_pred):
    """
    Returns RMSE and R² score for regression model evaluation.
    """
    try:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "R2_Score": r2}

    except Exception as e:
        raise CustomException(e, sys)

