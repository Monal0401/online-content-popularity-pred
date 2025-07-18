import os
import logging
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from joblib import dump, load

# ===========================
# 🔹 LOGGING SETUP
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===========================
# 🔹 PATHS
# ===========================
DATA_PATH = "notebook/data/OnlineNewsPopularity.csv"
SCALER_PATH = "artifacts/scaler.pkl"
REG_MODEL_PATH = "artifacts/reg_model.pkl"
CLF_MODEL_PATH = "artifacts/clf_model.pkl"

# ===========================
# 🔹 GLOBAL VARIABLES
# ===========================
scaler, reg_model, clf_model, df = None, None, None, None

# ===========================
# 🔹 TRAIN OR LOAD MODELS
# ===========================
def train_models(force_retrain=False):
    global scaler, reg_model, clf_model, df

    logger.info("✅ Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    logger.info(f"📊 Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    # ✅ Pick correct target column
    if "shares" in df.columns:
        target_col = "shares"
    else:
        possible_targets = [col for col in df.columns if "share" in col.lower()]
        if not possible_targets:
            raise ValueError("❌ Could not find any column containing 'shares' in the dataset!")
        target_col = possible_targets[-1]  # fallback
    logger.info(f"✅ Using '{target_col}' as the target column.")

    # ✅ Create popularity class
    df["popular"] = (df[target_col] > df[target_col].median()).astype(int)

    # ✅ Feature selection
    feature_cols = [
        "n_tokens_content",
        "num_hrefs",
        "num_imgs",
        "average_token_length",
        "global_subjectivity",
        "global_rate_positive_words"
    ]

    # ✅ Validate required features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing required features: {missing_features}")

    X = df[feature_cols]
    y_reg = df[target_col]
    y_clf = df["popular"]

    # ✅ Retrain if required
    if force_retrain or not (os.path.exists(SCALER_PATH) and os.path.exists(REG_MODEL_PATH) and os.path.exists(CLF_MODEL_PATH)):
        logger.info("🔄 Training new models...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_model.fit(X_scaled, y_reg)

        clf_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        clf_model.fit(X_scaled, y_clf)

        # ✅ Save models
        os.makedirs("artifacts", exist_ok=True)
        dump(scaler, SCALER_PATH)
        dump(reg_model, REG_MODEL_PATH)
        dump(clf_model, CLF_MODEL_PATH)
        logger.info("✅ Models trained & saved successfully!")
        return "✅ Models retrained successfully!"

    else:
        logger.info("✅ Loading saved models instead of retraining...")
        scaler = load(SCALER_PATH)
        reg_model = load(REG_MODEL_PATH)
        clf_model = load(CLF_MODEL_PATH)
        return "✅ Loaded saved models!"

# ===========================
# 🔹 PREDICTION FUNCTION
# ===========================
def predict_shares_and_popularity(n_tokens_content, num_hrefs, num_imgs, average_token_length, global_subjectivity, global_rate_positive_words):
    try:
        input_data = np.array([[n_tokens_content, num_hrefs, num_imgs, average_token_length, global_subjectivity, global_rate_positive_words]])
        input_scaled = scaler.transform(input_data)

        shares_pred = int(reg_model.predict(input_scaled)[0])
        popular_pred = clf_model.predict(input_scaled)[0]
        label = "🔥 Popular" if popular_pred == 1 else "📉 Not Popular"

        # ✅ Compare with dataset avg & median
        avg_shares = int(df["shares"].mean())
        med_shares = int(df["shares"].median())

        output_text = f"📈 Predicted Shares: {shares_pred}\n\n🎯 Popularity Prediction: {label}"

        # ✅ Plot comparison
        labels = ["Predicted", "Average", "Median"]
        values = [shares_pred, avg_shares, med_shares]
        colors = ["#4caf50", "#2196f3", "#ff9800"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 200, f"{yval:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Number of Shares")
        ax.set_title("Predicted vs Average & Median Shares")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        return output_text, fig

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return f"❌ Error: {str(e)}", None

# ===========================
# 🔹 RETRAIN BUTTON FUNCTION
# ===========================
def retrain_button_action():
    msg = train_models(force_retrain=True)
    return msg

# ===========================
# 🔹 GRADIO UI
# ===========================
logger.info("✅ App started successfully!")
startup_msg = train_models(force_retrain=False)

with gr.Blocks() as demo:
    gr.Markdown("## 📊 Online Content Popularity Dashboard")
    gr.Markdown("Enter article features to predict **shares & popularity** using Random Forest + XGBoost")

    with gr.Row():
        n_tokens_content = gr.Number(label="Number of Tokens in Content")
        num_hrefs = gr.Number(label="Number of Hyperlinks")
        num_imgs = gr.Number(label="Number of Images")

    with gr.Row():
        average_token_length = gr.Number(label="Average Token Length")
        global_subjectivity = gr.Number(label="Global Subjectivity")
        global_rate_positive_words = gr.Number(label="Global Rate of Positive Words")

    output_text = gr.Textbox(label="Prediction Results")
    output_plot = gr.Plot(label="Comparison Chart")

    predict_btn = gr.Button("🔮 Predict Popularity")
    retrain_btn = gr.Button("🔄 Reset & Retrain Models")
    retrain_status = gr.Textbox(label="Retrain Status", interactive=False)

    predict_btn.click(
        predict_shares_and_popularity,
        inputs=[n_tokens_content, num_hrefs, num_imgs, average_token_length, global_subjectivity, global_rate_positive_words],
        outputs=[output_text, output_plot]
    )

    retrain_btn.click(
        retrain_button_action,
        inputs=[],
        outputs=retrain_status
    )

# ===========================
# 🔹 LAUNCH APP
# ===========================
demo.launch()
