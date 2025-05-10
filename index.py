import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load the original dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Add extra manually-labeled resumes to improve training
extra_resumes = [
    {
        "Category": "Software Engineer",
        "Resume": """Experienced software engineer with 3+ years of experience in full-stack web development. Proficient in React, Node.js, PostgreSQL, and AWS. Built scalable applications in agile teams and contributed to CI/CD pipelines."""
    },
    {
        "Category": "Software Engineer",
        "Resume": """Full-stack engineer specializing in TypeScript, Express.js, and cloud deployment. Created internal tools for startups, improved backend performance, and mentored junior devs."""
    },
    {
        "Category": "Python Developer",
        "Resume": """Developed REST APIs using Flask and FastAPI. Wrote data processing pipelines with Pandas and Numpy. Deployed machine learning models with Docker and AWS Lambda."""
    },
    {
        "Category": "Web Designing",
        "Resume": """Creative web designer with a passion for UX/UI. Designed responsive websites using HTML, CSS, JavaScript, and Figma. Created landing pages and collaborated with frontend teams."""
    },
    {
        "Category": "HR",
        "Resume": """HR specialist with 5+ years of experience in recruiting, employee relations, and onboarding. Proficient in HRIS systems, benefits administration, and performance management."""
    },
]

# Convert to DataFrame and append to original
extra_df = pd.DataFrame(extra_resumes)
df = pd.concat([df, extra_df], ignore_index=True)

print("✅ Extra resumes added. Updated dataset shape:", df.shape)


# Look at the structure of the dataset
print(df.head())
print(df.columns)
print(df.info())
print(df["Category"].value_counts())

# === START NLP CLEANING ===

# Download required NLTK resources (do this only once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)        # Remove non-letters
    text = text.lower()                           # Lowercase
    tokens = nltk.word_tokenize(text)             # Tokenize
    tokens = [lemmatizer.lemmatize(word) 
              for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return ' '.join(tokens)

# Apply cleaning function to the resumes
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

# Show cleaned text
print(df[['Resume', 'Cleaned_Resume']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)  # You can adjust this later

# Transform the cleaned resume text into feature vectors
X = tfidf.fit_transform(df['Cleaned_Resume']).toarray()

# Target labels (what we want to predict)
y = df['Category']

# Check shape of data
print("TF-IDF Matrix Shape:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

import joblib

# Save the trained model
joblib.dump(model, "resume_classifier_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
