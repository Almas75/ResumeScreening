# RecruitAI – AI Powered Resume Screening System

RecruitAI is an AI-powered resume screening and candidate ranking system that helps recruiters automatically analyze resumes, match them with job descriptions, and identify the best candidates using Machine Learning.

The system reduces manual effort in recruitment by providing intelligent resume analysis, ATS compatibility checks, and candidate ranking.

---

## Features

### 1. Candidate Filtering

* Upload a CSV dataset containing candidate information
* Train machine learning models automatically
* Rank candidates based on similarity with the job description
* Display top N candidates using cosine similarity scoring

### 2. ATS Resume Checker

* Upload resumes in **PDF, DOCX, or TXT format**
* Compare resume content with the job description
* Calculate **ATS match percentage**
* Identify **missing skills and keywords**
* Provide **improvement suggestions**

### 3. Skill Gap Analysis

The system categorizes missing keywords into:

* Technical Skills
* Tools & Technologies
* Soft Skills
* Other Role-Specific Keywords

### 4. Learning Resource Recommendations

If skills are missing, the system suggests resources such as:

* Coursera
* freeCodeCamp
* Udemy
* Official Documentation

### 5. Multiple Machine Learning Models

The system evaluates several models and selects the best one:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

---

## Technologies Used

### Backend

* Python
* Flask
* Scikit-learn
* Pandas
* Joblib
* PDFPlumber
* Python-docx

### Frontend

* HTML
* CSS
* JavaScript

### Machine Learning

* TF-IDF Vectorization
* Cosine Similarity
* Classification Models

---

## Project Structure

```
RecruitAI
│
├── backend
│   ├── app.py
│   ├── model.pkl
│   ├── tfidf.pkl
│   └── label_encoder.pkl
│
├── frontend
│   ├── index.html
│   ├── ats.html
│   ├── style.css
│   ├── script.js
│   └── sample_candidates.csv
│
└── README.md

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/recruitai.git
```

### 2. Navigate to the project folder

```
cd recruitai
```

### 3. Install dependencies

```
pip install flask flask-cors scikit-learn pandas joblib pdfplumber python-docx
```

### 4. Run the backend server

```
python app.py
```

The application will start at:

```
http://localhost:5000
```

---

## How to Use

### Candidate Ranking

1. Upload the candidate dataset (CSV file)
2. Paste the job description
3. Choose number of candidates to display
4. Click **Train & Get Candidates**

### ATS Resume Checker

1. Upload a resume file
2. Paste the job description
3. Click **Generate Matching Score**
4. View ATS score, missing skills, and improvement tips

---

## Sample Dataset Format

The CSV dataset should include columns like:

```
Name
Skills
Experience (Years)
Education
Certifications
Job Role
Salary Expectation ($)
Projects Count
AI Score (0-100)
Recruiter Decision
```

---

## Example Use Cases

* Automated resume screening
* Applicant Tracking System (ATS) analysis
* Candidate ranking for recruitment
* Resume skill gap detection
* HR automation tools

---

## Future Improvements

* Deep learning based resume matching
* Semantic search using Sentence Transformers
* Resume scoring visualization
* Dashboard analytics for recruiters
* Integration with LinkedIn job posts

---

## License

This project is for educational and research purposes.
