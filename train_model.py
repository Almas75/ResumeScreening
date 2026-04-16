import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

# Load data
df = pd.read_csv("backend/AI_Resume_Screening.csv")
df.columns = df.columns.str.strip()

df["resume_text"] = (
    df["Skills"].fillna("") + " " +
    df["Education"].fillna("") + " " +
    df["Certifications"].fillna("") + " " +
    df["Job Role"].fillna("")
).apply(clean_text)

# Train model
tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
X = tfidf.fit_transform(df["resume_text"])
le = LabelEncoder()
y = le.fit_transform(df["Recruiter Decision"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=44)

model = RandomForestClassifier(n_estimators=200, random_state=44)
model.fit(X_tr, y_tr)

# Save models
joblib.dump(model, "backend/model.pkl")
joblib.dump(tfidf, "backend/tfidf.pkl")
joblib.dump(le, "backend/label_encoder.pkl")

print("Models saved successfully!")