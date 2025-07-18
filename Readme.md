
# 📊 Predicting Online Content Popularity

## ✨ Quick Summary

This project predicts the **popularity of online news articles** by:
✅ Estimating **how many times an article will be shared** (Regression)
✅ Classifying whether an article is **Popular 🔥 or Not Popular 📉** (Classification)
✅ Providing an **interactive Gradio dashboard** for live predictions

It automates **data ingestion → preprocessing → model training → evaluation → deployment**.

## ℹ️ About the Project

The **Online Content Popularity Prediction** project aims to analyze online news articles and determine their **potential popularity**. Using **machine learning models**, it predicts:

* **Continuous popularity** → exact number of shares (regression task)
* **Categorical popularity** → whether the article will be popular or not (classification task)

This helps **content creators, digital marketers, and media companies** make **data-driven decisions** about publishing strategies.

The project follows a **modular approach**, with separate pipelines for:

* **Data ingestion & cleaning**
* **Feature engineering & transformation**
* **Model training & hyperparameter tuning**
* **Deployment as an interactive app**

## 🛠 Tools & Technologies Used

* **Python 3.10** → Core programming language
* **Pandas & NumPy** → Data processing & numerical computations
* **Scikit-learn** → ML preprocessing, regression, and classification models
* **XGBoost** → Boosted models for better performance
* **SMOTE** → Handling imbalanced datasets
* **Matplotlib & Seaborn** → Exploratory Data Analysis (EDA) visualization
* **Gradio** → Interactive web-based prediction dashboard
* **Git & GitHub** → Version control and project hosting

## 📂 Dataset

* **Name:** Online News Popularity
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
* **Rows:** 39,644 | **Columns:** 61
* **Target Variables:**

  * `shares` → total number of shares
  * `popular` → binary label (1 if shares > median, else 0)

**Features include:**

* **Text-related** (token counts, average word length)
* **Multimedia** (images, videos)
* **Sentiment** (subjectivity, polarity)
* **Timing** (weekday publishing)
* **Content categories** (lifestyle, tech, world, etc.)

## 🛠 Workflow

1️⃣ **Data Loading & Cleaning**
2️⃣ **Feature Engineering** → binary popularity label, one-hot encoding
3️⃣ **Preprocessing** → scaling, handling imbalance with SMOTE
4️⃣ **EDA** → visualizing correlations & distributions
5️⃣ **Model Training**

* **Regression:** Linear Regression, Random Forest, XGBoost
* **Classification:** Random Forest Classifier, XGBoost Classifier
  6️⃣ **Hyperparameter Tuning** with `GridSearchCV`
  7️⃣ **Deployment** → Gradio-based prediction dashboard

## 🖥 Gradio Dashboard

The interactive **Gradio app** allows you to:
✅ Input article features
✅ Get **predicted number of shares**
✅ Get **popularity classification (Popular / Not Popular)**
✅ Compare predicted shares vs average & median

## 🔮 Future Improvements

* Add **deep learning models (LSTM, Transformers)**
* Extract **text embeddings** for better NLP features
* Deploy on **Hugging Face Spaces** or **AWS**
* Add **real-time web scraping** for predictions

## Installation

Clone repo & install dependencies:
```bash
git clone https://github.com/Monal0401/online-content-popularity-pred.git
cd online-content-popularity-pred
pip install -r requirements.txt
```

Run the training pipeline:
```bash
python -m src.pipeline.train_pipeline
```

Launch the dashboard:
```bash
python app.py
```

## 📚 References

* [Online News Popularity Dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Gradio](https://www.gradio.app/)


## 👤 Author

**Monal**
🔗 [GitHub Profile](https://github.com/Monal0401)



