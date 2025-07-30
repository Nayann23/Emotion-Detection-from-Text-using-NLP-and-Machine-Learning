import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Emotion index to label + emoji
emotion_map = {
    0: '😄 Joy',
    1: '😢 Sadness',
    2: '😠 Anger',
    3: '😱 Fear',
    4: '🤢 Disgust',
    5: '😮 Surprise',
    6: '❤️ Love'
}

# Function to clean and preprocess user input
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ''.join([ch for ch in text if not ch.isdigit()])           # remove digits
    text = ''.join([ch for ch in text if ch.isascii()])               # remove emojis
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in words if word not in stop_words]      # remove stopwords
    return ' '.join(cleaned)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="centered")

# Header
st.markdown("""
    <div style='text-align: center; padding: 10px'>
        <h1 style='color: #4CAF50;'>🧠 Emotion Detection App</h1>
        <p style='font-size: 18px;'>Type any sentence and I’ll guess the emotion behind it!</p>
    </div>
""", unsafe_allow_html=True)

# Input box
user_input = st.text_area("💬 Type your sentence here:", height=150)

# Prediction
if st.button("🔍 Predict Emotion", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        emotion = emotion_map.get(prediction, "Unknown")

        # Display the result
        st.markdown(f"<h2 style='text-align:center; color:#2196F3;'>Predicted Emotion: {emotion}</h2>", unsafe_allow_html=True)



# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
