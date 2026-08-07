import streamlit as st
import pickle

# Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article to check its authenticity")

# User Input
news_text = st.text_area("Paste the news content here:")

if st.button("Predict"):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Transform text using the vectorizer
        transformed_text = vectorizer.transform([news_text])
        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 1:
            st.error("❌ This news is likely **FAKE**.")
        else:
            st.success("✅ This news seems to be **REAL**.")