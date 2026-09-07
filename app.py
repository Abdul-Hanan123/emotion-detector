import streamlit as st
import joblib
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@st.cache_resource
def ensure_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))


STOP_WORDS = ensure_nltk_data()


def clean_text(text):
    """Must exactly mirror the preprocessing used during training."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    words = word_tokenize(text)
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)


# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🌿",
    layout="centered"
)

# ==============================================================
# EMOTION IDENTITY — muted colors + emoji
# ==============================================================
EMOTION_STYLE = {
    "joy":      {"color": "#D9A441", "emoji": "😄"},
    "sadness":  {"color": "#4C6A8C", "emoji": "😢"},
    "anger":    {"color": "#B24A3D", "emoji": "😠"},
    "love":     {"color": "#C06B82", "emoji": "❤️"},
    "fear":     {"color": "#6E5A8A", "emoji": "😨"},
    "surprise": {"color": "#3E8F82", "emoji": "😲"},
}
DEFAULT_STYLE = {"color": "#8A8A78", "emoji": "🤔"}

# ==============================================================
# GLOBAL CSS — sage background, indigo accent
# ==============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background-color: #E9EDE5;
}

.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2B3A2E;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: #66735F;
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}

.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #2B3A2E !important;
    border: 1px solid #D3DACB !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border: 1px solid #3A4A6B !important;
    box-shadow: 0 0 0 1px #3A4A6B !important;
}
.stTextArea textarea::placeholder {
    color: #A4AC9A !important;
}

.stButton button {
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    padding: 0.55rem 0;
    border: 1px solid transparent;
}
button[kind="primary"] {
    background-color: #3A4A6B !important;
    color: #FFFFFF !important;
}
button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #56634F !important;
    border: 1px solid #D3DACB !important;
}

section[data-testid="stSidebar"] {
    background-color: #DFE5D8;
    border-right: 1px solid #CBD3C0;
}
section[data-testid="stSidebar"] * {
    color: #4E5B48 !important;
    font-family: 'Manrope', sans-serif !important;
}
section[data-testid="stSidebar"] h3 {
    color: #2B3A2E !important;
    font-weight: 700;
}

/* Result card */
.result-card {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    background-color: #FFFFFF;
    border: 1px solid #D3DACB;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin: 1.6rem 0 1.4rem 0;
}
.result-emoji {
    font-size: 2.2rem;
    line-height: 1;
}
.result-text {
    font-size: 1.25rem;
    font-weight: 700;
    text-transform: capitalize;
}
.result-sub {
    font-size: 0.85rem;
    color: #8A9080;
    margin-top: 2px;
}

/* Confidence rows */
.conf-heading {
    font-size: 0.78rem;
    font-weight: 600;
    color: #7C8574;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: 9px;
}
.conf-emoji {
    width: 22px;
    font-size: 0.95rem;
}
.conf-label {
    width: 76px;
    font-size: 0.88rem;
    color: #4E5B48;
    text-transform: capitalize;
}
.conf-track {
    flex: 1;
    background-color: #E3E8DC;
    border-radius: 5px;
    height: 8px;
    overflow: hidden;
    margin: 0 10px;
}
.conf-fill {
    height: 100%;
    border-radius: 5px;
}
.conf-pct {
    width: 42px;
    text-align: right;
    font-size: 0.8rem;
    color: #7C8574;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# LOAD MODEL
# ==============================================================
@st.cache_resource
def load_model():
    model = joblib.load("emotion_model.pkl")
    emotion_numbers = joblib.load("emotion_labels.pkl")
    number_to_emotion = {v: k for k, v in emotion_numbers.items()}
    return model, number_to_emotion

try:
    model, number_to_emotion = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ==============================================================
# HERO
# ==============================================================
st.markdown('<div class="hero-title">🌿 Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Enter a sentence and the model will classify the emotion.</div>',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(
        "Model files not found. Make sure `emotion_model.pkl` and "
        "`emotion_labels.pkl` are in the same folder as this app."
    )
    st.stop()

# ==============================================================
# INPUT
# ==============================================================
# Widget state stored under key "user_input" so Clear can actually reset it
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "",
    placeholder="I can't believe I got the job, I'm so excited...",
    height=120,
    label_visibility="collapsed",
    key="user_input"
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.user_input = ""
    st.rerun()

# ==============================================================
# PREDICTION
# ==============================================================
if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        predicted_emotion = number_to_emotion[prediction]
        style = EMOTION_STYLE.get(predicted_emotion, DEFAULT_STYLE)

        # --- Result card ---
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{style['emoji']}</div>
            <div>
                <div class="result-text" style="color:{style['color']};">{predicted_emotion}</div>
                <div class="result-sub">predicted emotion</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Confidence breakdown ---
        # Use model.classes_ (not raw range/index) so the emotion labels
        # always line up correctly with the probability/score columns,
        # even if class order isn't sequential.
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned_input])[0]
            classes = model.classes_
            conf_df = pd.DataFrame({
                "emotion": [number_to_emotion[c] for c in classes],
                "value": probs
            })
            heading = "Confidence (calibrated)"
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function([cleaned_input])[0]
            confidences = softmax(raw_scores)
            classes = model.classes_
            conf_df = pd.DataFrame({
                "emotion": [number_to_emotion[c] for c in classes],
                "value": confidences
            })
            heading = "Confidence (relative, not calibrated)"
        else:
            conf_df = None
            heading = None

        if conf_df is not None:
            conf_df = conf_df.sort_values("value", ascending=False)
            st.markdown(f'<div class="conf-heading">{heading}</div>', unsafe_allow_html=True)
            rows_html = ""
            for _, row in conf_df.iterrows():
                emo = row["emotion"]
                pct = row["value"] * 100
                emo_style = EMOTION_STYLE.get(emo, DEFAULT_STYLE)
                rows_html += f"""
                <div class="conf-row">
                    <div class="conf-emoji">{emo_style['emoji']}</div>
                    <div class="conf-label">{emo}</div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{pct:.1f}%; background:{emo_style['color']};"></div>
                    </div>
                    <div class="conf-pct">{pct:.1f}%</div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("### About")
    st.write(
        "A calibrated SVM model trained on TF-IDF text features, "
        "classifying text into one of six emotions."
    )
    st.markdown("**Test accuracy:** ~89%")