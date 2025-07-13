import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion for Online News Popularity project")

        try:
            # Step 1: Load dataset
            df = pd.read_csv("notebook/data/OnlineNewsPopularity.csv")
            df.columns = df.columns.str.strip()  # Remove whitespace
            logging.info("Dataset loaded successfully")

            # Step 2: Remove top 5% outliers based on 'shares'
            original_len = len(df)
            df = df[df['shares'] < df['shares'].quantile(0.95)]
            logging.info(f"Removed top 5% outliers. Rows reduced from {original_len} to {len(df)}")

            # Step 3: Create binary label 'popular'
            df['popular'] = (df['shares'] > df['shares'].median()).astype(int)

            # Step 4: Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Step 5: Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred in data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")


