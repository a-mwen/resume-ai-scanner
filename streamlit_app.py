import streamlit as st
import joblib
import nltk
import re
import fitz  # PyMuPDF for PDFs
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

import os

nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")

# Only download if not already present
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, 'corpora/wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaner
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# --- UI START ---
st.title("🧠 AI Resume Screener")
st.write("Upload a resume file or paste your resume below to predict the best job fit.")

# Paste input
manual_text = st.text_area("📄 Option 1: Paste your resume", height=250)

# File upload input
uploaded_file = st.file_uploader("📤 Option 2: Or upload a resume file", type=["pdf", "txt"])

# Extracted text logic
resume_text = ""

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
elif manual_text.strip():
    resume_text = manual_text

# Only predict if there's input
if resume_text.strip():
    if st.button("Predict"):
        cleaned_input = clean_text(resume_text)
        input_vector = vectorizer.transform([cleaned_input])
        prediction_proba = model.predict_proba(input_vector)[0]

        # Get top 3 predictions
        top_indices = prediction_proba.argsort()[-3:][::-1]
        top_categories = model.classes_[top_indices]
        top_confidences = prediction_proba[top_indices]

        st.success("✅ Top Predicted Job Categories:")
        for category, confidence in zip(top_categories, top_confidences):
            st.write(f"**{category}** — {round(confidence * 100, 2)}%")
