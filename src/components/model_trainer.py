import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier, XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')


@dataclass
class ModelTrainingConfig:
    cls_model_path: str = os.path.join("artifacts", "best_classification_model.pkl")
    reg_model_path: str = os.path.join("artifacts", "best_regression_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test data for Classification & Regression")

            X_train = train_array[:, :-2]
            y_train_reg = train_array[:, -2]
            y_train_cls = train_array[:, -1]

            X_test = test_array[:, :-2]
            y_test_reg = test_array[:, -2]
            y_test_cls = test_array[:, -1]

            # ✅ Handle imbalanced classification using SMOTE
            smote = SMOTE(random_state=42)
            X_train_cls, y_train_cls_balanced = smote.fit_resample(X_train, y_train_cls)
            logging.info(f"After SMOTE, y_train_cls distribution: {dict(pd.Series(y_train_cls_balanced).value_counts())}")

            # -------------------- CLASSIFICATION MODEL WITH GRID SEARCH --------------------
            logging.info("Starting hyperparameter tuning for XGBoostClassifier...")

            xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            }

            grid_search = GridSearchCV(
                xgb,
                param_grid,
                scoring="accuracy",
                cv=3,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train_cls, y_train_cls_balanced)

            best_xgb = grid_search.best_estimator_
            logging.info(f"Best params for XGBClassifier: {grid_search.best_params_}")

            # ✅ Extra: 5-Fold Cross-Validation for classification
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cls_cv_scores = cross_val_score(best_xgb, X_train_cls, y_train_cls_balanced, cv=kf, scoring="accuracy")
            logging.info(f"Classification 5-Fold CV Accuracy: {cls_cv_scores.mean():.4f} ± {cls_cv_scores.std():.4f}")

            # Evaluate on holdout test set
            y_pred_cls = best_xgb.predict(X_test)
            cls_acc = accuracy_score(y_test_cls, y_pred_cls)
            f1 = f1_score(y_test_cls, y_pred_cls, average="weighted")
            try:
                auc = roc_auc_score(y_test_cls, best_xgb.predict_proba(X_test), multi_class='ovr')
            except Exception:
                logging.warning("Could not calculate ROC AUC score for classification.")
                auc = 0.0

            logging.info(f"Tuned XGBClassifier → Test Accuracy={cls_acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

            # -------------------- REGRESSION MODELS --------------------
            reg_models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
                "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42),
            }

            reg_best_score = -999
            reg_best_model_name = None
            reg_best_model = None

            for name, model in reg_models.items():
                logging.info(f"Training Regression model: {name}")

                # ✅ Extra: Cross-validation for regression
                reg_cv_scores = cross_val_score(model, X_train, y_train_reg, cv=kf, scoring="r2")
                logging.info(f"{name} 5-Fold CV R²: {reg_cv_scores.mean():.4f} ± {reg_cv_scores.std():.4f}")

                model.fit(X_train, y_train_reg)
                y_pred_reg = model.predict(X_test)

                r2 = r2_score(y_test_reg, y_pred_reg)
                rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

                logging.info(f"{name} → Test R²={r2:.3f}, RMSE={rmse:.2f}")

                if r2 > reg_best_score:
                    reg_best_score = r2
                    reg_best_model_name = name
                    reg_best_model = model

            if reg_best_score < 0.4:
                raise CustomException(f"No good regression model found. Best R²={reg_best_score:.3f}")

            logging.info(f"Best Regression Model: {reg_best_model_name} (R²={reg_best_score:.3f})")
            logging.info(f"Best Classification Model: XGBoostClassifier (Accuracy={cls_acc:.3f})")

            # Save models
            save_object(self.model_trainer_config.cls_model_path, best_xgb)
            save_object(self.model_trainer_config.reg_model_path, reg_best_model)

            logging.info(f"Saved classification model → {self.model_trainer_config.cls_model_path}")
            logging.info(f"Saved regression model → {self.model_trainer_config.reg_model_path}")

            return reg_best_score, cls_acc

        except Exception as e:
            raise CustomException(e, sys)
