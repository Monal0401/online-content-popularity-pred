import os
import sys

from src.logger import logging
from src.exception import CustomException

# Import project components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    try:
        logging.info("Starting Online Content Popularity Training Pipeline...")

        # DATA INGESTION (Reads CSV from notebook/data)
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(f"✅ Data Ingestion completed: Train={train_data_path}, Test={test_data_path}")

        # DATA TRANSFORMATION (Scaling + preprocessing)
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path
        )
        logging.info("Data Transformation completed successfully.")

        # MODEL TRAINING (Regression + Classification)
        trainer = ModelTrainer()
        reg_r2, clf_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        # Log and print final results
        logging.info(f"Training completed! Regression R²={reg_r2:.4f}, Classification Accuracy={clf_accuracy:.4f}")
        print(f"\n Final Results:\n Regression R² Score: {reg_r2:.4f}\n Classification Accuracy: {clf_accuracy:.4f}")

    except Exception as e:
        logging.error("Training pipeline failed due to an error.")
        logging.error(f"Exception details: {e}")  # Add this line to log error detail
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
