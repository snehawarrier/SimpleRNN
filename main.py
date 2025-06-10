# main.py — Streamlit Sentiment App for IMDB Reviews

import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

# ------------------------------
# Load IMDB Word Index
# ------------------------------
word_index = imdb.get_word_index()

# ------------------------------
# Load Pretrained Model
# ------------------------------
model = load_model('simple_rnn_imdb_fixed.h5')  # make sure this model is retrained!

# ------------------------------
# Helper Function to Preprocess Input
# ------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 = <UNK>
    padded = pad_sequences([encoded], maxlen=500)
    return padded

# ------------------------------
# Streamlit UI
# ------------------------------
st.title('🎬 IMDB Movie Review Sentiment Classifier')
st.write("Enter a movie review, and I'll predict whether it's **Positive** or **Negative**.")

# User Input
user_input = st.text_area('✏️ Write your movie review here:')

if st.button('🔍 Predict Sentiment'):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        score = prediction[0][0]
        sentiment = 'Positive 😊' if score >= 0.5 else 'Negative 😞'

        # Display result
        st.subheader("📢 Prediction Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {score:.4f}")
else:
    st.info("👆 Enter a review and click the button to classify sentiment.")
