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

# List of common English stop words to filter out
common_words = {"and", "the", "for", "with", "this", "that", "from", "your", "will", "our", "are", "have", "been", "was", "were", "but", "not", "can", "they", "then", "into", "has", "more", "now", "well", "only", "about", "also", "some", "when", "into", "where", "how", "all", "any", "each", "few", "most", "other", "such", "than", "very", "should", "could"}

# A more comprehensive list of technical keywords to help extraction
TECHNICAL_SKILLS_LIST = [
    "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust",
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring",
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle", "nosql",
    "aws", "azure", "google cloud", "docker", "kubernetes", "jenkins", "terraform", "ansible",
    "machine learning", "ml", "deep learning", "neural network", "neural networks", "llm", "llms", 
    "large language model", "large language models", "nlp", "natural language processing", "computer vision",
    "data science", "data analysis", "tableau", "power bi", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "gitlab", "bitbucket", "linux", "unix", "bash", "agile", "scrum", "devops",
    "rest api", "graphql", "microservices", "unit testing", "integration testing", "docker", "kubernetes"
]

# Map synonyms/acronyms to a single canonical name to avoid duplicates
TECHNICAL_SYNONYMS = {
    "ml": "machine learning",
    "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "nlp": "natural language processing",
    "neural network": "neural networks",
}

def clean_text(text):
    """Standard cleaning for ML model prediction (retains original behavior)"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def clean_technical_text(text):
    """Cleaning for technical keyword extraction (preserves terms like C++, .NET)"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # Remove only non-technical punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-]', ' ', text)
    return re.sub(r"\s+", " ", text)

def extract_text(file):
    try:
        # Save current position in case it was read before
        file.seek(0)
        if file.filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                text = "".join([page.extract_text() or "" for page in pdf.pages])
                return text
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.filename.endswith('.txt'):
            return file.read().decode('utf-8')
    except Exception as e:
        print(f"Error extracting text: {e}")
    return ""

@app.route("/")
def index():
    return send_from_directory(frontend_dir, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(frontend_dir, path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = f"{data['skills']} {data['education']} {data['certifications']} {data['job_role']}"
        vector = tfidf.transform([clean_text(text)])
        prediction = model.predict(vector)[0]
        decision = le.inverse_transform([prediction])[0]
        return jsonify({"decision": decision})
    except NameError:
        return jsonify({"error": "Model files not loaded properly"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    try:
        resume_file = request.files.get('resume')
        job_desc = request.form.get('job_desc', '')
        
        if not resume_file:
            return jsonify({"error": "No resume file uploaded"}), 400
            
        resume_text = extract_text(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from the resume"}), 400

        # Technical keyword extraction with deduplication
        job_desc_cleaned = clean_technical_text(job_desc)
        resume_text_cleaned = clean_technical_text(resume_text)
        
        # 1. Identify which technical concepts are in the JD
        found_canonical_skills = set()
        for skill in TECHNICAL_SKILLS_LIST:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, job_desc_cleaned):
                canonical = TECHNICAL_SYNONYMS.get(skill.lower(), skill)
                found_canonical_skills.add(canonical)
        
        # 2. Check resume for these canonical skills (checking all synonyms)
        matched = []
        missing = []
        
        # Reverse map for resume checking (Canonical -> list of all possible terms)
        def get_all_variants(canonical_name):
            variants = {canonical_name}
            for syn, canon in TECHNICAL_SYNONYMS.items():
                if canon == canonical_name:
                    variants.add(syn)
            return variants

        for canonical in found_canonical_skills:
            variants = get_all_variants(canonical)
            is_found = False
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, resume_text_cleaned):
                    is_found = True
                    break
            
            if is_found:
                matched.append(canonical)
            else:
                missing.append(canonical)
        
        matched = sorted(matched)
        missing = sorted(missing)
        
        total_req = len(found_canonical_skills)
        match_percent = (len(matched) / total_req * 100) if total_req > 0 else 0

        # UI Color Coding Logic
        if match_percent >= 80:
            level, color = "Excellent Match", "#059669"
        elif match_percent >= 60:
            level, color = "Good Match", "#d97706"
        else:
            level, color = "Needs Improvement", "#dc2626"

        improvement_tips = []
        if matched:
            improvement_tips.append(f"Great! Your resume includes key technical terms like {', '.join(matched[:5])}.")
        if missing:
            improvement_tips.append(f"Consider adding technical keywords such as {', '.join(missing[:5])} to better match the job description.")
        
        if not found_canonical_skills:
            improvement_tips.append("No specific technical skills were identified from the job description. Try adding more technical requirements.")

        learning_resources = []
        if missing:
            learning_resources = [
                {"platform": "Coursera", "url": "https://www.coursera.org", "description": "University-backed certifications and specialized courses."},
                {"platform": "Udemy", "url": "https://www.udemy.com", "description": "Hands-on projects and practical skill-building tutorials."},
                {"platform": "Documentation & Labs", "url": "https://docs.microsoft.com", "description": "Free interactive learning paths and technical documentation."},
                {"platform": "LinkedIn Learning", "url": "https://www.linkedin.com/learning", "description": "Professional courses to boost your technical and soft skills."}
            ]

        return jsonify({
            "match_percent": round(match_percent, 2),
            "match_level": level,
            "level_color": color,
            "matched_skills": matched,
            "total_matched": len(matched),
            "total_required": total_req,
            "missing_categories": {"technical": missing},
            "total_missing": len(missing),
            "recommendation": "Focus on incorporating the key technical terms from the job description into your resume.",
            "improvement_tips": improvement_tips,
            "learning_resources": learning_resources
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/train_and_candidates", methods=["POST"])
def train_and_candidates():
    try:
        csv_file = request.files.get('csv')
        job_desc = request.form.get('job_desc', '')
        num_candidates = int(request.form.get('num_candidates', 10))

        # Read CSV data
        df = pd.read_csv(csv_file)

        # Check required columns
        required_columns = ['Skills', 'Education', 'Certifications', 'Job Role', 'Recruiter Decision']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {', '.join(missing_columns)}"}), 400

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