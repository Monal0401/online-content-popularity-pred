import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

class PredictPipeline:
    def __init__(self):
        # Paths for saved models & preprocessor
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.cls_model_path = os.path.join("artifacts", "best_classification_model.pkl")
        self.reg_model_path = os.path.join("artifacts", "best_regression_model.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Takes a DataFrame of new data and returns both regression & classification predictions.
        """
        try:
            print("✅ Loading preprocessor and models...")
            preprocessor = load_object(self.preprocessor_path)
            cls_model = load_object(self.cls_model_path)
            reg_model = load_object(self.reg_model_path)
            print("✅ Successfully loaded all models")

            # Transform input data using the same preprocessor
            print("🔄 Transforming input features...")
            transformed_data = preprocessor.transform(features)

            # Predict classification (popularity class)
            print("📊 Predicting popularity class...")
            cls_preds = cls_model.predict(transformed_data)

            # Predict regression (shares/log_shares)
            print("📈 Predicting shares (regression)...")
            reg_preds = reg_model.predict(transformed_data)

            return {
                "predicted_popularity_class": cls_preds,
                "predicted_shares": reg_preds
            }

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class to structure new input data for prediction.
    """

    def __init__(
        self,
        n_tokens_title: int,
        n_tokens_content: int,
        n_unique_tokens: float,
        num_hrefs: int,
        num_self_hrefs: int,
        num_imgs: int,
        num_videos: int,
        average_token_length: float,
        kw_min_min: float,
        kw_max_min: float,
        kw_avg_min: float,
        kw_min_max: float,
        kw_max_max: float,
        kw_avg_max: float,
        kw_min_avg: float,
        kw_max_avg: float,
        kw_avg_avg: float,
        self_reference_min_shares: float,
        self_reference_max_shares: float,
        self_reference_avg_sharess: float,
        weekday_is_monday: int,
        weekday_is_tuesday: int,
        weekday_is_wednesday: int,
        weekday_is_thursday: int,
        weekday_is_friday: int,
        weekday_is_saturday: int,
        weekday_is_sunday: int,
        is_weekend: int,
    ):
        # Map all inputs
        self.n_tokens_title = n_tokens_title
        self.n_tokens_content = n_tokens_content
        self.n_unique_tokens = n_unique_tokens
        self.num_hrefs = num_hrefs
        self.num_self_hrefs = num_self_hrefs
        self.num_imgs = num_imgs
        self.num_videos = num_videos
        self.average_token_length = average_token_length
        self.kw_min_min = kw_min_min
        self.kw_max_min = kw_max_min
        self.kw_avg_min = kw_avg_min
        self.kw_min_max = kw_min_max
        self.kw_max_max = kw_max_max
        self.kw_avg_max = kw_avg_max
        self.kw_min_avg = kw_min_avg
        self.kw_max_avg = kw_max_avg
        self.kw_avg_avg = kw_avg_avg
        self.self_reference_min_shares = self_reference_min_shares
        self.self_reference_max_shares = self_reference_max_shares
        self.self_reference_avg_sharess = self_reference_avg_sharess
        self.weekday_is_monday = weekday_is_monday
        self.weekday_is_tuesday = weekday_is_tuesday
        self.weekday_is_wednesday = weekday_is_wednesday
        self.weekday_is_thursday = weekday_is_thursday
        self.weekday_is_friday = weekday_is_friday
        self.weekday_is_saturday = weekday_is_saturday
        self.weekday_is_sunday = weekday_is_sunday
        self.is_weekend = is_weekend

    def get_data_as_data_frame(self):
        """
        Converts the input data into a pandas DataFrame for prediction.
        """
        try:
            input_dict = {
                "n_tokens_title": [self.n_tokens_title],
                "n_tokens_content": [self.n_tokens_content],
                "n_unique_tokens": [self.n_unique_tokens],
                "num_hrefs": [self.num_hrefs],
                "num_self_hrefs": [self.num_self_hrefs],
                "num_imgs": [self.num_imgs],
                "num_videos": [self.num_videos],
                "average_token_length": [self.average_token_length],
                "kw_min_min": [self.kw_min_min],
                "kw_max_min": [self.kw_max_min],
                "kw_avg_min": [self.kw_avg_min],
                "kw_min_max": [self.kw_min_max],
                "kw_max_max": [self.kw_max_max],
                "kw_avg_max": [self.kw_avg_max],
                "kw_min_avg": [self.kw_min_avg],
                "kw_max_avg": [self.kw_max_avg],
                "kw_avg_avg": [self.kw_avg_avg],
                "self_reference_min_shares": [self.self_reference_min_shares],
                "self_reference_max_shares": [self.self_reference_max_shares],
                "self_reference_avg_sharess": [self.self_reference_avg_sharess],
                "weekday_is_monday": [self.weekday_is_monday],
                "weekday_is_tuesday": [self.weekday_is_tuesday],
                "weekday_is_wednesday": [self.weekday_is_wednesday],
                "weekday_is_thursday": [self.weekday_is_thursday],
                "weekday_is_friday": [self.weekday_is_friday],
                "weekday_is_saturday": [self.weekday_is_saturday],
                "weekday_is_sunday": [self.weekday_is_sunday],
                "is_weekend": [self.is_weekend],
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)
