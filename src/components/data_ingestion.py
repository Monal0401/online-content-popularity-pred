import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
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
        logging.info("🔄 Starting Data Ingestion process.")
        try:
            # Load original dataset
            df = pd.read_csv('notebook/data/OnlineNewsPopularity.csv')
            logging.info("✅ Dataset loaded successfully.")

            # Strip spaces in column names
            df.columns = df.columns.str.strip()

            # Drop unnecessary columns if present
            df = df.drop(columns=['url', 'timedelta'], errors='ignore')

            # Replace 'TRUE'/'FALSE' with 1/0
            df.replace({'TRUE': 1, 'FALSE': 0}, inplace=True)

            # Ensure all data is numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            df.fillna(0, inplace=True)

            # Create log_shares column
            df['log_shares'] = np.log1p(df['shares'])

            # Create classification target
            def classify_popularity(shares):
                if shares < 1400:
                    return 0
                elif shares < 5000:
                    return 1
                else:
                    return 2
            df['popularity_class'] = df['shares'].apply(classify_popularity)

            # Ensure directory exists, then save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"📦 Raw data saved → {self.ingestion_config.raw_data_path}")

            # Split into train/test (70/30 split)
            train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

            # Ensure directory exists, then save train data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Ensure directory exists, then save test data
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"✅ Train-test split (70/30) complete. Data saved in {os.path.dirname(self.ingestion_config.raw_data_path)}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("❌ Error occurred in data ingestion.")
            raise CustomException(e, sys)

# Optional: Run it standalone
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print("Train path:", train_data_path)
    print("Test path:", test_data_path)
