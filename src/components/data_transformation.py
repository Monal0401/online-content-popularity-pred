import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, features: pd.DataFrame) -> ColumnTransformer:
        try:
            numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()

            # If publish_day is present (one-hot encoded), remove from numerical
            numerical_cols = [col for col in numerical_cols if not col.startswith("weekday_is_")]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([
                        ("scaler", StandardScaler())
                    ]), numerical_cols)
                ],
                remainder='passthrough'  # to keep one-hot encoded day columns
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded training and testing datasets")

            # Drop URL column if present
            train_df.drop(columns=['url'], errors='ignore', inplace=True)
            test_df.drop(columns=['url'], errors='ignore', inplace=True)

            # Derive publish_day from weekday columns
            if 'weekday_is_monday' in train_df.columns:
                for df in [train_df, test_df]:
                    df['publish_day'] = df.loc[:, 'weekday_is_monday':'weekday_is_sunday'].idxmax(axis=1)
                    df['publish_day'] = df['publish_day'].apply(lambda x: x.split('_')[-1])
                    df.drop(columns=df.loc[:, 'weekday_is_monday':'weekday_is_sunday'].columns, inplace=True)

            # One-hot encode publish_day
            train_df = pd.get_dummies(train_df, columns=['publish_day'], drop_first=True)
            test_df = pd.get_dummies(test_df, columns=['publish_day'], drop_first=True)

            # Align train/test columns after one-hot encoding
            train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

            # Store target columns
            target_reg = 'shares'
            target_clf = 'popular'

            # Drop targets from input
            input_features_train = train_df.drop(columns=[target_reg, target_clf])
            input_features_test = test_df.drop(columns=[target_reg, target_clf])

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object(input_features_train)

            # Fit-transform training data and transform test data
            input_train_scaled = preprocessing_obj.fit_transform(input_features_train)
            input_test_scaled = preprocessing_obj.transform(input_features_test)

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Output arrays for model training
            train_arr = {
                'X': input_train_scaled,
                'y_reg': train_df[target_reg].values,
                'y_clf': train_df[target_clf].values
            }
            test_arr = {
                'X': input_test_scaled,
                'y_reg': test_df[target_reg].values,
                'y_clf': test_df[target_clf].values
            }

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
