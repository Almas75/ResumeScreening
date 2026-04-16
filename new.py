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
# GLOBAL STYLES  (warm light, editorial)
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Nunito:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #f7f4ef;
    --surface:   #ffffff;
    --surface2:  #fdf9f4;
    --border:    #e8e0d4;
    --accent:    #e8541a;
    --accent2:   #2563d4;
    --accent3:   #059669;
    --warn:      #d97706;
    --danger:    #dc2626;
    --text:      #1a1612;
    --text2:     #4a4540;
    --muted:     #9d958c;
    --radius:    16px;
    --shadow:    0 2px 20px rgba(26,22,18,0.07);
    --shadow-lg: 0 8px 48px rgba(26,22,18,0.12);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }

/* ── Hero ── */
.hero {
    background: var(--surface);
    border-radius: 24px;
    padding: 52px 56px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border);
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 460px; height: 100%;
    background: linear-gradient(135deg, #fff9f5 0%, #ffecd9 50%, #ffe0c8 100%);
    clip-path: polygon(20% 0%, 100% 0%, 100% 100%, 0% 100%);
    z-index: 0;
}
.hero::after {
    content: '✦';
    position: absolute;
    top: 28px; right: 44px;
    font-size: 110px;
    color: rgba(232,84,26,0.09);
    font-family: serif;
    line-height: 1;
    z-index: 1;
}
.hero-inner { position: relative; z-index: 2; }
.hero-tag {
    display: inline-block;
    background: #fff3ee;
    border: 1px solid #fcd0b8;
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 18px;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.4rem, 4.5vw, 3.8rem) !important;
    font-weight: 900 !important;
    line-height: 1.08 !important;
    color: var(--text) !important;
    margin: 0 0 18px !important;
    letter-spacing: -1.5px;
}
.hero h1 em { font-style: italic; color: var(--accent); }
.hero p {
    font-size: 15px;
    font-weight: 400;
    color: var(--text2);
    max-width: 420px;
    line-height: 1.75;
    margin: 0 0 28px;
}
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; }
.stat-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text2);
}
.stat-pill .icon {
    width: 24px; height: 24px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}
.stat-pill .icon.orange { background: #fff0e8; }
.stat-pill .icon.blue   { background: #eef2ff; }
.stat-pill .icon.green  { background: #ecfdf5; }

/* ── Section heading ── */
.sec-head { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.sec-head-bar { width: 4px; height: 22px; background: var(--accent); border-radius: 2px; }
.sec-head-text { font-family: 'Playfair Display', serif; font-size: 16px; font-weight: 700; color: var(--text); }

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }

/* ── Accuracy rows ── */
.acc-row {
    display: flex;
    align-items: center;
    padding: 11px 0;
    border-bottom: 1px solid var(--border);
    gap: 12px;
    font-size: 13.5px;
}
.acc-row:last-child { border-bottom: none; }
.acc-name { color: var(--text2); min-width: 170px; font-weight: 500; }
.acc-track { flex: 1; height: 6px; background: #f0ece6; border-radius: 3px; }
.acc-fill  { height: 100%; border-radius: 3px; }
.acc-pct   { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 15px; min-width: 52px; text-align: right; }
.best-badge {
    background: #fff3ee; color: var(--accent);
    border: 1px solid #fcd0b8;
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px; border-radius: 999px;
}

/* ── Best banner ── */
.best-banner {
    background: linear-gradient(135deg, #fff8f5, #fff0e8);
    border: 1px solid #fcd0b8;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 14px; font-weight: 600;
    color: var(--accent);
    display: flex; align-items: center; gap: 10px;
    margin: 16px 0 24px;
}

/* ── Score card ── */
.score-wrap {
    display: flex; align-items: center; gap: 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
}
.score-circle {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    flex-shrink: 0;
    border: 3px solid;
}
.score-circle.ex { border-color: var(--accent3); background: #f0fdf8; }
.score-circle.gd { border-color: var(--warn);    background: #fffbeb; }
.score-circle.pr { border-color: var(--danger);  background: #fff5f5; }
.score-num { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 900; line-height: 1; }
.score-pct { font-size: 10px; font-weight: 600; letter-spacing: 1px; opacity: 0.7; }
.score-info h3 { font-family: 'Playfair Display', serif; font-size: 20px; font-weight: 700; margin: 0 0 6px; }
.score-info p  { font-size: 13px; color: var(--text2); margin: 0; line-height: 1.6; }

/* ── Chips ── */
.chip-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip { padding: 5px 13px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.chip.match   { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.chip.missing { background: #fff5f5; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Divider ── */
.hdivider { border: none; border-top: 1px solid var(--border); margin: 28px 0; }

/* ── Warn box ── */
.warn-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 13.5px;
    font-weight: 500;
    color: #92400e;
}

/* ── Footer ── */
.footer { text-align: center; padding: 32px 0 12px; font-size: 12px; color: var(--muted); letter-spacing: 0.5px; }
.footer strong { color: var(--accent); }

/* ── Streamlit overrides ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Nunito', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    color: var(--muted) !important;
    padding: 10px 28px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 30px !important;
    gap: 0 !important;
}
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    padding: 10px 30px !important;
    box-shadow: 0 4px 16px rgba(232,84,26,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    background: #d04310 !important;
    box-shadow: 0 6px 24px rgba(232,84,26,0.35) !important;
}
[data-testid="stDownloadButton"] > button {
    background: #fff !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    box-shadow: none !important;
}
textarea {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stFileUploader"] > div {
    background: var(--surface2) !important;
    border: 1.5px dashed #d4c9bb !important;
    border-radius: 14px !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
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
      <div class="stat-pill"><span class="icon orange">🔍</span> Semantic Ranking</div>
      <div class="stat-pill"><span class="icon blue">🤖</span> 5 ML Models</div>
      <div class="stat-pill"><span class="icon green">📋</span> ATS Skill Audit</div>
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


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs(["🔍  Candidate Filter", "📊  ATS Resume Checker"])


# ═════════════════════════════════════════════
# TAB 1
# ═════════════════════════════════════════════

with tab1:

    col_l, col_r = st.columns([1.15, 1], gap="large")

    with col_l:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-bar"></div>
          <div class="sec-head-text">Resume Dataset</div></div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV with candidate data", type=["csv"])

    with col_r:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-bar" style="background:#2563d4"></div>
          <div class="sec-head-text">Job Description</div></div>""", unsafe_allow_html=True)
        job_desc = st.text_area(
            "Paste the job description",
            height=150, key="jd1"
        )
        num_candidates = st.slider("Top candidates to show", 5, 50, 10, 5)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
        st.markdown("""<div class="sec-head">
          <div class="sec-head-bar" style="background:#059669"></div>
          <div class="sec-head-text">Dataset Preview</div></div>""", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=360)

        df["resume_text"] = (
            df["Skills"].fillna("") + " " +
            df["Education"].fillna("") + " " +
            df["Certifications"].fillna("") + " " +
            df["Job Role"].fillna("")
        ).apply(clean_text)

        st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

        if st.button("✦  Train Models & Find Best Candidates"):

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

            accs = {}
            best_model, best_acc = None, 0
            for name, m in ml_models.items():
                m.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, m.predict(X_te))
                accs[name] = acc
                if acc > best_acc:
                    best_acc, best_model = acc, m

            st.markdown("""<div class="sec-head" style="margin-top:8px">
              <div class="sec-head-bar"></div>
              <div class="sec-head-text">Model Performance</div></div>""", unsafe_allow_html=True)

            COLORS = ["#e8541a","#2563d4","#059669","#d97706","#7c3aed"]
            rows_html = ""
            for i, (name, acc) in enumerate(accs.items()):
                pct = round(acc * 100, 2)
                is_best = (acc == best_acc)
                badge = '<span class="best-badge">Best</span>' if is_best else ""
                color = COLORS[i % len(COLORS)]
                rows_html += f"""
                <div class="acc-row">
                  <span class="acc-name">{name} {badge}</span>
                  <div class="acc-track"><div class="acc-fill" style="width:{pct}%;background:{color}"></div></div>
                  <span class="acc-pct" style="color:{color}">{pct}%</span>
                </div>"""

            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="best-banner">🏆 Best model accuracy: {round(best_acc*100,2)}% — used for candidate ranking</div>',
                unsafe_allow_html=True
            )

            jv = tfidf.transform([clean_text(job_desc)])
            df["Match Score"] = cosine_similarity(X, jv).flatten()
            result = df.sort_values("Match Score", ascending=False)

            st.markdown("""<div class="sec-head">
              <div class="sec-head-bar" style="background:#2563d4"></div>
              <div class="sec-head-text">Top Matching Candidates</div></div>""", unsafe_allow_html=True)
            st.dataframe(result.head(num_candidates).style.format({"Match Score": "{:.3f}"}), use_container_width=True)
            st.download_button("⬇  Download as CSV", result.to_csv(index=False), file_name="matched_candidates.csv")


# ═════════════════════════════════════════════
# TAB 2
# ═════════════════════════════════════════════

with tab2:

    col_a, col_b = st.columns([1, 1.1], gap="large")

    with col_a:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-bar"></div>
          <div class="sec-head-text">Upload Resume</div></div>""", unsafe_allow_html=True)
        resume_file = st.file_uploader("PDF, DOCX or TXT accepted", type=["pdf","docx","txt"])

    with col_b:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-bar" style="background:#2563d4"></div>
          <div class="sec-head-text">Job Description</div></div>""", unsafe_allow_html=True)
        job_desc2 = st.text_area(
            "Paste the job description",
            height=185, key="jd2",
            placeholder="Paste the role requirements, skills, and responsibilities here…"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✦  Analyse My Resume"):

        if resume_file and job_desc2.strip():

            resume_text = extract_text(resume_file)
            emb = sentence_model.encode([resume_text, job_desc2])
            sim = cosine_similarity(emb[0].reshape(1,-1), emb[1].reshape(1,-1))[0][0]
            score = round(sim * 100, 2)

            if score >= 80:
                cls, label, col = "ex", "Excellent Match ✦", "#059669"
            elif score >= 60:
                cls, label, col = "gd", "Good Match", "#d97706"
            else:
                cls, label, col = "pr", "Needs Improvement", "#dc2626"

            st.markdown(f"""
            <div class="score-wrap">
              <div class="score-circle {cls}">
                <span class="score-num" style="color:{col}">{score}</span>
                <span class="score-pct" style="color:{col}">% MATCH</span>
              </div>
              <div class="score-info">
                <h3 style="color:{col}">{label}</h3>
                <p>Semantic alignment between your resume and the job description,<br>
                   measured using sentence-level embeddings.</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            rw = {w for w in re.findall(r"\b\w+\b", resume_text.lower()) if is_technical(w)}
            jw = {w for w in re.findall(r"\b\w+\b", job_desc2.lower()) if is_technical(w)}
            matched = sorted(jw & rw)
            missing = sorted(jw - rw)

            c1, c2 = st.columns(2, gap="medium")

            with c1:
                st.markdown("""<div class="sec-head">
                  <div class="sec-head-bar" style="background:#059669"></div>
                  <div class="sec-head-text">Matched Skills</div></div>""", unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if matched:
                    chips = "".join(f'<span class="chip match">{w}</span>' for w in matched[:22])
                    st.markdown(f'<div class="chip-grid">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#9d958c;font-size:13px;margin:0">No matched keywords found.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown("""<div class="sec-head">
                  <div class="sec-head-bar" style="background:#dc2626"></div>
                  <div class="sec-head-text">Missing Skills</div></div>""", unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if missing:
                    chips = "".join(f'<span class="chip missing">{w}</span>' for w in missing[:22])
                    st.markdown(f'<div class="chip-grid">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#059669;font-size:13px;font-weight:600;margin:0">✓ All key skills are present!</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="warn-box">
              ⚠ Please upload a resume <strong>and</strong> enter a job description before running the analysis.
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<hr class="hdivider">
<div class="footer">
  Built with <strong>♥</strong> using Streamlit · Sentence Transformers · scikit-learn
</div>
""", unsafe_allow_html=True)