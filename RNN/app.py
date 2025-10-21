import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import numpy as np

# Load model
model = tf.keras.models.load_model("RNN/sentiment_rnn_model.keras")

# Load IMDB word index
word_index = imdb.get_word_index()

# Parameters
max_len = 500

# Function to preprocess user input
def preprocess_text(text):
    tokens = text_to_word_sequence(text)
    encoded = [word_index.get(word, 2) + 3 for word in tokens if word in word_index]  # +3 offset as IMDB does
    padded = pad_sequences([encoded], maxlen=max_len)
    return padded

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Predictor")
st.write("Enter a movie review and see if the model thinks it’s **positive** or **negative**!")

user_input = st.text_area("✍️ Type your movie review here:", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        data = preprocess_text(user_input)
        prediction = model.predict(data)[0][0]
        sentiment = "😊 Positive" if prediction >= 0.5 else "😞 Negative"
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: {prediction:.2f}")
