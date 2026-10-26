import pickle
from pathlib import Path

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "spam_classifier.pkl"

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    initial_sidebar_state="auto",
)

st.title("Spam or Ham Message Classifier")

if not MODEL_PATH.is_file():
    st.error("Trained model not found. Run the cells in train_model.ipynb first.")
    st.stop()

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

message = st.text_area("Enter a message to classify:", height=150)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message.lower()])[0]
        if prediction == "spam":
            st.error("🚫 This message is classified as **SPAM**")
        else:
            st.success("✅ This message is classified as **HAM** (Not Spam)")
