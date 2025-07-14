import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create preprocessing pipeline for numerical columns
        """
        try:
            # All features in your dataset are numerical after cleaning
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, slice(0, -2))  # exclude last 2 target columns
                ],
                remainder="passthrough"  # for log_shares and popularity_class
            )

            logging.info("✅ Preprocessor pipeline created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train/test sets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("📄 Train and test datasets loaded.")

            # Separate features and target
            target_column = "log_shares"
            classification_column = "popularity_class"

            X_train = train_df.drop(columns=["shares", target_column, classification_column], errors="ignore")
            y_train = train_df[target_column]
            y_train_cls = train_df[classification_column]

            X_test = test_df.drop(columns=["shares", target_column, classification_column], errors="ignore")
            y_test = test_df[target_column]
            y_test_cls = test_df[classification_column]

            # Build and apply preprocessor
            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("✅ Data transformation applied.")

            # Combine features with both targets (you can choose which to use later)
            train_arr = np.c_[
                X_train_transformed, y_train.to_numpy().reshape(-1, 1), y_train_cls.to_numpy().reshape(-1, 1)
            ]
            test_arr = np.c_[
                X_test_transformed, y_test.to_numpy().reshape(-1, 1), y_test_cls.to_numpy().reshape(-1, 1)
            ]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("💾 Preprocessing object saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
