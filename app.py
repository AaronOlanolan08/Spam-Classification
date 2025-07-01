import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and clean the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# Split and train
X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    initial_sidebar_state="auto"        
)
# Streamlit UI
st.title("Spam or Ham Message Classifier")

message = st.text_area("Enter a message to classify:", height=150)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([message.lower()])
        prediction = model.predict(input_vec)[0]
        if prediction == "spam":
            st.error("🚫 This message is classified as **SPAM**")
        else:
            st.success("✅ This message is classified as **HAM** (Not Spam)")
