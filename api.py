from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Flask app setup
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLP cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === Endpoint: Predict Job Category ===
@app.route('/predict-role', methods=['POST'])
def predict_role():
    data = request.get_json()
    resume_text = data.get('resume', '')
    if not resume_text:
        return jsonify({"error": "Resume text is required"}), 400

    cleaned = clean_text(resume_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = round(100 * max(proba), 2)

    return jsonify({
        "predicted_category": prediction,
        "confidence": f"{confidence}%"
    })

# === Endpoint: Match Resume to Job Description ===
@app.route('/match-job', methods=['POST'])
def match_job():
    data = request.get_json()
    resume_text = data.get('resume', '')
    job_desc = data.get('job_description', '')
    if not resume_text or not job_desc:
        return jsonify({"error": "Both resume and job_description are required"}), 400

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_desc)

    vectors = vectorizer.transform([cleaned_resume, cleaned_job])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    score = round(similarity * 100, 2)

    return jsonify({
        "match_score": f"{score}%",
        "similarity_value": similarity
    })

# === Run the API ===
if __name__ == '__main__':
    app.run(debug=True)
