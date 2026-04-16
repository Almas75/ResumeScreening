from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pdfplumber
import docx
import os

app = Flask(__name__)
CORS(app)

# Resolve path to frontend folder
frontend_dir = os.path.abspath("../frontend")

# Load pre-trained models (Ensure these files exist in the backend folder)
try:
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    le = joblib.load("label_encoder.pkl")
except:
    print("Warning: .pkl files not found. Predict route may fail.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def extract_text(file):
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
            return text
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    return ""

@app.route("/")
def index():
    return send_from_directory(frontend_dir, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(frontend_dir, path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = f"{data['skills']} {data['education']} {data['certifications']} {data['job_role']}"
    vector = tfidf.transform([clean_text(text)])
    prediction = model.predict(vector)[0]
    decision = le.inverse_transform([prediction])[0]
    return jsonify({"decision": decision})

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    try:
        resume_file = request.files.get('resume')
        job_desc = request.form.get('job_desc', '')
        resume_text = extract_text(resume_file)
        
        resume_words = set(re.findall(r"\b\w+\b", resume_text.lower()))
        job_words = set(re.findall(r"\b\w+\b", job_desc.lower()))
        
        matched = sorted(list(job_words & resume_words))
        missing = sorted(list(job_words - resume_words))
        match_percent = len(matched) / len(job_words) * 100 if job_words else 0

        # UI Color Coding Logic
        if match_percent >= 80:
            level, color = "Excellent Match", "#059669"
        elif match_percent >= 60:
            level, color = "Good Match", "#d97706"
        else:
            level, color = "Needs Improvement", "#dc2626"

        # Categorize Gaps
        tech_keywords = ['python', 'java', 'sql', 'react', 'aws', 'docker', 'ml', 'api', 'tensorflow', 'pytorch', 'html', 'css', 'javascript']
        tool_keywords = ['react', 'aws', 'docker', 'kubernetes', 'git', 'github', 'tableau', 'excel', 'jira', 'salesforce', 'powerbi']
        soft_keywords = ['communication', 'teamwork', 'leadership', 'problem', 'collaboration', 'adaptability', 'organized', 'creativity', 'time', 'reliable']

        resource_map = {
            'python': ['Coursera: Python for Everybody', 'Real Python', 'Codecademy Python'],
            'java': ['Udemy: Java Programming Masterclass', 'Coursera: Java Programming and Software Engineering Fundamentals'],
            'sql': ['Mode Analytics SQL Tutorial', 'Khan Academy SQL'],
            'react': ['freeCodeCamp React', 'Scrimba Learn React'],
            'aws': ['AWS Skill Builder', 'Coursera AWS Fundamentals'],
            'docker': ['Docker Docs', 'Udemy Docker Mastery'],
            'kubernetes': ['Kubernetes Basics by Google', 'Udemy Kubernetes Certified'],
            'ml': ['Coursera Machine Learning by Andrew Ng', 'fast.ai Practical Deep Learning'],
            'api': ['Postman API Fundamentals', 'Udemy REST API Design'],
            'html': ['freeCodeCamp HTML Course', 'MDN Web Docs'],
            'css': ['freeCodeCamp CSS Course', 'MDN Web Docs'],
            'javascript': ['freeCodeCamp JavaScript Algorithms', 'Eloquent JavaScript'],
            'tensorflow': ['TensorFlow Developer Certificate', 'Coursera TensorFlow in Practice'],
            'pytorch': ['DeepLearning.AI PyTorch', 'Udemy PyTorch for Deep Learning'],
            'github': ['GitHub Learning Lab', 'freeCodeCamp Git & GitHub'],
            'tableau': ['Tableau Public Resources', 'Udemy Tableau Bootcamp'],
            'excel': ['Excel Easy', 'Coursera Excel Skills for Business'],
            'jira': ['Atlassian Jira Tutorials', 'Udemy Jira Essentials'],
            'powerbi': ['Microsoft Learn Power BI', 'Udemy Power BI A-Z']
        }

        technical = [s for s in missing if s in tech_keywords]
        tools = [s for s in missing if s in tool_keywords and s not in technical]
        soft = [s for s in missing if s in soft_keywords]
        other = [s for s in missing if s not in technical + tools + soft][:10]

        resources = []
        for keyword in missing:
            if keyword in resource_map:
                for resource in resource_map[keyword]:
                    if resource not in resources:
                        resources.append(resource)

        improvement_tips = []
        if technical:
            improvement_tips.append(f"Add technical keywords like {', '.join(technical[:5])} to your resume and support them with concrete project examples.")
        if tools:
            improvement_tips.append(f"Highlight tool experience with {', '.join(tools[:5])}, including hands-on projects or certifications.")
        if soft:
            improvement_tips.append(f"Showcase soft skills such as {', '.join(soft[:5])} through achievements and teamwork examples.")
        if other:
            improvement_tips.append(f"Include more role-specific keywords such as {', '.join(other[:5])} from the job description.")
        if not missing:
            improvement_tips.append("Your resume already matches the job description well. Keep the language specific and results-focused.")

        return jsonify({
            "match_percent": round(match_percent, 2),
            "match_level": level,
            "level_color": color,
            "matched_skills": matched[:15],
            "total_matched": len(matched),
            "total_required": len(job_words),
            "missing_categories": {"technical": technical, "tools": tools, "soft_skills": soft, "other": other},
            "total_missing": len(missing),
            "recommendation": "Tailor your resume by adding the missing keywords found below.",
            "improvement_tips": improvement_tips,
            "learning_resources": resources[:8]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train_and_candidates", methods=["POST"])
def train_and_candidates():
    try:
        csv_file = request.files.get('csv')
        job_desc = request.form.get('job_desc', '')
        num_candidates = int(request.form.get('num_candidates', 10))

        # Read CSV data
        df = pd.read_csv(csv_file)

        # Prepare training data
        features = []
        for _, row in df.iterrows():
            text = f"{row.get('Skills', '')} {row.get('Education', '')} {row.get('Certifications', '')} {row.get('Job Role', '')}"
            features.append(clean_text(text))

        # Convert to TF-IDF
        X = tfidf.transform(features)
        y = le.transform(df['Recruiter Decision'].values)

        # Train multiple models and find the best
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42)
        }

        best_accuracy = 0
        best_model = None

        for name, model in models.items():
            model.fit(X, y)
            accuracy = accuracy_score(y, model.predict(X))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        # Calculate match scores for all candidates
        job_vector = tfidf.transform([clean_text(job_desc)])
        similarities = cosine_similarity(X, job_vector).flatten()

        # Add match scores to dataframe
        df_copy = df.copy()
        df_copy['Match Score'] = similarities

        # Get top candidates
        top_candidates = df_copy.nlargest(num_candidates, 'Match Score')

        # Utility to safely convert candidate values to JSON-friendly output
        def safe_value(candidate, key):
            if key not in candidate.index:
                return 'N/A'
            value = candidate[key]
            return 'N/A' if pd.isna(value) else value

        # Convert to list of dictionaries
        candidates = []
        for _, candidate in top_candidates.iterrows():
            candidates.append({
                'Name': safe_value(candidate, 'Name'),
                'Skills': safe_value(candidate, 'Skills'),
                'Experience (Years)': safe_value(candidate, 'Experience (Years)'),
                'Education': safe_value(candidate, 'Education'),
                'Certifications': safe_value(candidate, 'Certifications'),
                'Job Role': safe_value(candidate, 'Job Role'),
                'Salary Expectation ($)': safe_value(candidate, 'Salary Expectation ($)'),
                'Projects Count': safe_value(candidate, 'Projects Count'),
                'AI Score (0-100)': safe_value(candidate, 'AI Score (0-100)'),
                'Recruiter Decision': safe_value(candidate, 'Recruiter Decision'),
                'Match Score': float(candidate['Match Score'])
            })

        return jsonify({
            'best_accuracy': best_accuracy * 100,
            'candidates': candidates
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)