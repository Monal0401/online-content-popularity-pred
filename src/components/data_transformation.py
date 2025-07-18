import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Loading train and test datasets...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded.")

            # Drop target columns from features
            target_reg = 'log_shares'
            target_cls = 'popularity_class'

            # Features exclude targets
            features = [col for col in train_df.columns if col not in [target_reg, target_cls]]

            X_train = train_df[features]
            X_test = test_df[features]

            y_train_reg = train_df[target_reg].values
            y_train_cls = train_df[target_cls].values

            y_test_reg = test_df[target_reg].values
            y_test_cls = test_df[target_cls].values

            # Build preprocessing pipeline:
            # Impute missing values with mean and scale features
            # Use PolynomialFeatures degree=1 to avoid feature explosion (just pass features through)
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('poly', PolynomialFeatures(degree=1, include_bias=False)),  # No actual expansion, just pass-through
                ('scaler', StandardScaler())
            ])

            # Assuming all features are numeric here; if categorical, you can add categorical pipeline
            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, features)
            ])

            logging.info("Fitting and transforming training data...")
            X_train_processed = preprocessor.fit_transform(X_train)

            logging.info("Transforming test data...")
            X_test_processed = preprocessor.transform(X_test)

            # Combine features and targets into final numpy arrays
            # For model trainer: features + regression target + classification target (as last two columns)
            train_arr = np.c_[
                X_train_processed,
                y_train_reg,
                y_train_cls
            ]

            test_arr = np.c_[
                X_test_processed,
                y_test_reg,
                y_test_cls
            ]

            # Save preprocessing pipeline object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info(f"Preprocessing object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)
