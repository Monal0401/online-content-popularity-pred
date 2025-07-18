import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_regression(true, predicted):
    """
    Evaluate regression results with R2 and MSE.
    """
    try:
        r2 = r2_score(true, predicted)
        mse = mean_squared_error(true, predicted)
        return {"r2_score": r2, "mse": mse}
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification(true, predicted):
    """
    Evaluate classification results with accuracy, precision, recall, f1-score, and confusion matrix.
    """
    try:
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, average="weighted", zero_division=0)
        recall = recall_score(true, predicted, average="weighted", zero_division=0)
        f1 = f1_score(true, predicted, average="weighted", zero_division=0)
        cm = confusion_matrix(true, predicted)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
        }
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params, task_type="regression"):
    """
    Train multiple models with GridSearchCV and evaluate on train and test sets.
    task_type: "regression" or "classification" controls metrics and scoring.
    Returns a dict with model names as keys and test scores as values.
    """
    try:
        report = {}
        scoring = "neg_mean_squared_error" if task_type == "regression" else "accuracy"

        for model_name, model in models.items():
            try:
                print(f"\nTraining model: {model_name}")
                hyperparams = params.get(model_name, {})

                gs = GridSearchCV(
                    model,
                    hyperparams,
                    cv=3,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                    refit=True,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                if task_type == "regression":
                    train_metrics = evaluate_regression(y_train, y_train_pred)
                    test_metrics = evaluate_regression(y_test, y_test_pred)
                else:
                    train_metrics = evaluate_classification(y_train, y_train_pred)
                    test_metrics = evaluate_classification(y_test, y_test_pred)

                print(f"\n{model_name} Training Metrics:")
                for metric, value in train_metrics.items():
                    if metric != "confusion_matrix":
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}:\n{value}")

                print(f"\n{model_name} Testing Metrics:")
                for metric, value in test_metrics.items():
                    if metric != "confusion_matrix":
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}:\n{value}")

                # For reporting, use r2_score for regression or accuracy for classification
                report[model_name] = (
                    test_metrics["r2_score"] if task_type == "regression" else test_metrics["accuracy"]
                )

            except Exception as model_err:
                print(f"Skipping model '{model_name}' due to error: {model_err}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
