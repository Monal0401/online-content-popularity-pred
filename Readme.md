Online Content Popularity Prediction
Project Overview
This project predicts the popularity of online content using machine learning techniques. It performs both:

Regression: Predicting the log of the number of shares.

Classification: Categorizing content popularity into classes (low, medium, high).

The end-to-end pipeline includes data ingestion, preprocessing, model training with hyperparameter tuning, and evaluation.

File Structure
ONLINE_CONTENT_POPULARITY/
│
├── env/                         # Virtual environment (Python packages)
├── logs/                        # Log files for debugging/tracking
│
├── notebook/
│   ├── data/                    # Dataset folder (CSV files go here)
│   ├── eda.ipynb                # Exploratory Data Analysis notebook
│   └── model_training.ipynb     # Model training & evaluation notebook
│
├── src/                         # Source code
│   ├── __pycache__/             # Compiled Python cache files
│   │
│   ├── components/              # Data processing & ML components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/                # Training & prediction pipelines
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py             # Custom exception handling
│   ├── logger.py                # Logging utility
│   └── utils.py                 # Helper/utility functions
│
├── .gitignore                   # Git ignore rules
├── Readme.md                    # Project documentation
├── requirements.txt             # Dependencies list
└── setup.py                     # Installation setup script
