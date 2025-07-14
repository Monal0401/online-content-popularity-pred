import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save any Python object (like a model, preprocessor, scaler) to the specified file path using dill.
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
    Load a Python object from a file path using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(true, predicted):
    """
    Evaluate a regression model using R² score. Can be extended for other metrics.
    """
    try:
        score = r2_score(true, predicted)
        return score

    except Exception as e:
        raise CustomException(e, sys)
