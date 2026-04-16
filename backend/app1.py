import streamlit as st
import pandas as pd
import pdfplumber
import docx
import re
import string
import nltk

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

from nltk.corpus import words as nltk_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RecruitAI — Resume Screening",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #f7f4ef;
    --surface: #ffffff;
    --surface2: #fdf9f4;
    --border: #e8e0d4;
    --accent: #e8541a;
    --accent2: #2563d4;
    --accent3: #059669;
    --warn: #d97706;
    --danger: #dc2626;
    --text: #1a1612;
    --text2: #4a4540;
    --muted: #9d958c;
    --radius: 14px;
    --shadow: 0 2px 16px rgba(26,22,18,0.07);
    --shadow-lg: 0 8px 40px rgba(26,22,18,0.12);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }

/* ── Hero ── */
.hero {
    background: var(--surface);
    border-radius: 24px;
    padding: 48px 52px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-lg);
}
.hero::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 440px; height: 100%;
    background: linear-gradient(135deg, #fff9f5 0%, #ffecd9 55%, #ffe0c8 100%);
    clip-path: polygon(22% 0%, 100% 0%, 100% 100%, 0% 100%);
    z-index: 0;
}
.hero::after {
    content: '✦';
    position: absolute; top: 26px; right: 42px;
    font-size: 100px; color: rgba(232,84,26,0.08);
    font-family: serif; line-height: 1; z-index: 1; pointer-events: none;
}
.hero-inner { position: relative; z-index: 2; max-width: 500px; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fff3ee; border: 1px solid #fcd0b8; color: var(--accent);
    font-size: 10px; font-weight: 600; letter-spacing: 2.5px;
    text-transform: uppercase; padding: 5px 14px; border-radius: 999px; margin-bottom: 16px;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.2rem, 4vw, 3.4rem) !important;
    font-weight: 900 !important; line-height: 1.1 !important;
    letter-spacing: -1px; margin: 0 0 14px !important; color: var(--text) !important;
}
.hero h1 em { font-style: italic; color: var(--accent); }
.hero p {
    font-size: 14px; color: var(--text2); max-width: 420px;
    line-height: 1.8; margin: 0 0 26px; font-weight: 300;
}
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; }
.stat-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 7px 14px;
    font-size: 12px; font-weight: 500; color: var(--text2);
}
.stat-icon { width: 22px; height: 22px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 12px; }
.icon-org { background: #fff0e8; }
.icon-blue { background: #eef2ff; }
.icon-grn { background: #ecfdf5; }

/* ── Section heading ── */
.sec-head { display: flex; align-items: center; gap: 9px; margin-bottom: 12px; }
.sec-bar { width: 4px; height: 20px; border-radius: 2px; flex-shrink: 0; }
.sec-title { font-family: 'Playfair Display', serif; font-size: 15px; font-weight: 700; color: var(--text); }

/* ── Card ── */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 22px 26px; margin-bottom: 16px;
    box-shadow: var(--shadow); transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }

/* ── Accuracy rows ── */
.acc-row {
    display: flex; align-items: center; padding: 10px 0;
    border-bottom: 1px solid var(--border); gap: 12px; font-size: 13px;
}
.acc-row:last-child { border-bottom: none; }
.acc-name { color: var(--text2); min-width: 165px; font-weight: 500; }
.acc-track { flex: 1; height: 5px; background: #f0ece6; border-radius: 3px; }
.acc-fill { height: 100%; border-radius: 3px; }
.acc-pct { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 14px; min-width: 50px; text-align: right; }
.best-badge {
    background: #fff3ee; color: var(--accent); border: 1px solid #fcd0b8;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; padding: 2px 7px; border-radius: 999px;
}

/* ── Best banner ── */
.best-banner {
    background: linear-gradient(135deg, #fff8f5, #fff0e8);
    border: 1px solid #fcd0b8; border-radius: 10px;
    padding: 12px 18px; font-size: 13px; font-weight: 600; color: var(--accent);
    display: flex; align-items: center; gap: 10px; margin: 14px 0 22px;
}

/* ── Score card ── */
.score-wrap {
    display: flex; align-items: center; gap: 24px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 26px 30px;
    box-shadow: var(--shadow); margin-bottom: 22px;
}
.score-circle {
    width: 88px; height: 88px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; flex-shrink: 0; border: 3px solid;
}
.score-num { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 900; line-height: 1; }
.score-pct { font-size: 9px; font-weight: 600; letter-spacing: 1px; opacity: 0.7; }
.score-info h3 { font-family: 'Playfair Display', serif; font-size: 19px; font-weight: 700; margin: 0 0 6px; }
.score-info p { font-size: 12px; color: var(--text2); margin: 0; line-height: 1.6; }

/* ── Chips ── */
.chip-grid { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 8px; }
.chip { padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.chip-match { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.chip-miss  { background: #fff5f5; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Divider ── */
.hdivider { border: none; border-top: 1px solid var(--border); margin: 26px 0; }

/* ── Warn box ── */
.warn-box {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 10px; padding: 13px 18px;
    font-size: 13px; font-weight: 500; color: #92400e;
}

/* ── Footer ── */
.footer { text-align: center; padding: 28px 0 10px; font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }
.footer strong { color: var(--accent); }

/* ── Streamlit overrides ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important; color: var(--muted) !important;
    padding: 10px 28px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 28px !important; gap: 0 !important;
}

[data-testid="stButton"] > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 11px 28px !important;
    box-shadow: 0 4px 16px rgba(232,84,26,0.28) !important;
}
[data-testid="stButton"] > button:hover {
    background: #d04310 !important;
    box-shadow: 0 6px 24px rgba(232,84,26,0.38) !important;
}
[data-testid="stDownloadButton"] > button {
    background: #fff !important; color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important; box-shadow: none !important;
}

textarea {
    background: var(--surface2) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
}
textarea:focus { border-color: var(--accent) !important; }

[data-testid="stFileUploader"] > div {
    background: var(--surface2) !important;
    border: 1.5px dashed #d4c9bb !important; border-radius: 14px !important;
}
[data-testid="stFileUploader"] > div:hover { border-color: var(--accent) !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}

input[type="range"] { accent-color: var(--accent) !important; }

[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--accent) !important; color: #fff !important;
    border-radius: 999px !important; font-size: 11px !important; font-weight: 600 !important;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-tag">✦ AI-Powered Recruitment</div>
    <h1>Smart Hiring,<br><em>Simplified.</em></h1>
    <p>Screen hundreds of resumes in seconds using semantic AI matching, five ML classifiers, and automated ATS gap analysis.</p>
    <div class="stat-row">
      <div class="stat-pill"><span class="stat-icon icon-org">🔍</span> Semantic Ranking</div>
      <div class="stat-pill"><span class="stat-icon icon-blue">🤖</span> 5 ML Models</div>
      <div class="stat-pill"><span class="stat-icon icon-grn">📋</span> ATS Skill Audit</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sentence_model = load_sentence_model()

@st.cache_resource
def get_common_words():
    try:
    common = set(nltk_words.words())
    except:
    common = set()
    extra = {
    "is","the","a","an","and","or","in","on","at","to","for","of","with","by",
    "from","as","be","been","are","was","were","am","have","has","had","do",
    "does","did","will","would","should","could","can","may","might","must",
    "shall","if","that","this","it","its","your","you","we","they","he","she",
    "there","proficiency","like","languages","language","experience","skills",
    "skill","knowledge","ability","required","plus","preferred","understanding",
    "working","etc","years","year","month","months","high","level","strong",
    "good","excellent","proficient","experienced","role","team","work","job",
    "position","responsibilities","about","more","other","such","some","any",
    "all","own","get","make","give","know","think","use","people","way","new",
    "right","different","back","also","just","only","very","even"
    }
    return common.union(extra)

common_words = get_common_words()

def is_technical(w):
    return w not in common_words and len(w) > 2

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def extract_text(file):
    if file.type == "application/pdf":
    text = ""
    with pdfplumber.open(file) as pdf:
    for page in pdf.pages:
    text += page.extract_text() or ""
    return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])
    elif file.type == "text/plain":
    return file.read().decode("utf-8")
    return ""

def sec_head(title, color="var(--accent)"):
    return f"""<div class="sec-head">
    <div class="sec-bar" style="background:{color}"></div>
    <span class="sec-title">{title}</span>
    </div>"""

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# ═════════════════════════════════════════════
# CANDIDATE FILTER
# ═════════════════════════════════════════════
col_l, col_r = st.columns([1.15, 1], gap="large")

with col_l:
    st.markdown(sec_head("Resume Dataset"), unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
    "Upload CSV with candidate data", type=["csv"], label_visibility="collapsed"
    )

with col_r:
    st.markdown(sec_head("Job Description", "#2563d4"), unsafe_allow_html=True)
    job_desc = st.text_area(
    "Job description", height=150, key="jd1",
    placeholder="Paste the role requirements, skills, and responsibilities here…",
    label_visibility="collapsed"
    )
    num_candidates = st.slider("Top candidates to show", 5, 50, 10, 5)

    if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    st.markdown(sec_head("Dataset Preview", "#059669"), unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=360)

    # Build resume text column (gracefully handle missing columns)
    text_cols = ["Skills", "Education", "Certifications", "Job Role"]
    for col in text_cols:
    if col not in df.columns:
    df[col] = ""
    df["resume_text"] = (
    df["Skills"].fillna("") + " " +
    df["Education"].fillna("") + " " +
    df["Certifications"].fillna("") + " " +
    df["Job Role"].fillna("")
    ).apply(clean_text)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    if st.button("✦  Train Models & Find Best Candidates"):
        if "Recruiter Decision" not in df.columns:
            st.markdown(
                '<div class="warn-box">⚠ Column <strong>"Recruiter Decision"</strong> not found in the CSV. Please check your dataset.</div>',
                unsafe_allow_html=True
            )
        elif not job_desc.strip():
            st.markdown(
                '<div class="warn-box">⚠ Please enter a <strong>Job Description</strong> before training.</div>',
                unsafe_allow_html=True
            )
        else:
    with st.spinner("Training models…"):
        tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
        X = tfidf.fit_transform(df["resume_text"])
        le = LabelEncoder()
        y = le.fit_transform(df["Recruiter Decision"])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=44)

        ml_models = {
            "Logistic Regression":    LogisticRegression(max_iter=1000),
            "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=44),
            "AdaBoost":               AdaBoostClassifier(n_estimators=100, random_state=44),
            "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=44),
            "Support Vector Machine": SVC(),
        }
        COLORS = ["#e8541a", "#2563d4", "#059669", "#d97706", "#7c3aed"]

        accs = {}
        best_model, best_acc, best_name = None, 0, ""
        for name, m in ml_models.items():
            m.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, m.predict(X_te))
            accs[name] = acc
            if acc > best_acc:
                best_acc, best_model, best_name = acc, m, name

    # Model accuracy cards
    st.markdown(sec_head("Model Performance"), unsafe_allow_html=True)
    rows_html = ""
    for i, (name, acc) in enumerate(accs.items()):
        pct = round(acc * 100, 2)
        is_best = (name == best_name)
        badge = '<span class="best-badge">Best</span>' if is_best else ""
        color = COLORS[i % len(COLORS)]
        rows_html += f"""
        <div class="acc-row">
            <span class="acc-name">{name} {badge}</span>
            <div class="acc-track">
                <div class="acc-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <span class="acc-pct" style="color:{color}">{pct}%</span>
        </div>"""
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="best-banner">🏆 Best model: <strong>{best_name}</strong> — {round(best_acc*100,2)}% accuracy — used for candidate ranking</div>',
        unsafe_allow_html=True
    )

    # Semantic ranking
    jv = tfidf.transform([clean_text(job_desc)])
    df["Match Score"] = cosine_similarity(X, jv).flatten()
    result = df.sort_values("Match Score", ascending=False)

    st.markdown(sec_head("Top Matching Candidates", "#2563d4"), unsafe_allow_html=True)
    st.dataframe(
        result.head(num_candidates).style.format({"Match Score": "{:.3f}"}),
        use_container_width=True
    )
    st.download_button(
        "⬇  Download as CSV",
        result.to_csv(index=False),
        file_name="matched_candidates.csv"
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr class="hdivider">
<div class="footer">
  Built with <strong>♥</strong> using Streamlit · Sentence Transformers · scikit-learn
</div>
""", unsafe_allow_html=True)
