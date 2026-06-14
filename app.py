# ══════════════════════════════════════════════════════════════
#   STREAMLIT APP — Analyse de Sentiments Amazon
#   Groupe 8 | Pr. KAISS Wijdane
# ══════════════════════════════════════════════════════════════

import streamlit as st # streamlit run app.py
import joblib
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse de Sentiments",
    page_icon="/icon.png",
    layout="centered"
)

# ── Load model & vectorizer ───────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ── Preprocessing ─────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    nltk.download('punkt',     quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    return PorterStemmer(), set(stopwords.words('english'))

stemmer, stop_words = load_nlp()

def preprocess_text(text):
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ── UI ────────────────────────────────────────────────────────
st.title("Analyse de Sentiments — Avis Amazon")
st.markdown("**Projet Data Mining | Groupe 8 | Pr. KAISS Wijdane**")
st.markdown("---")

st.markdown("""
Ce modèle analyse le sentiment d'un avis client Amazon et prédit s'il est
**positif** ou **négatif**, entraîné sur **3,6 millions d'avis réels**.
""")

# ── Info boxes ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Modèle",    "Logistic Regression")
col2.metric("Accuracy",  "89.05%")
col3.metric("Dataset",   "3.6M avis")

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────
st.subheader("Entrez votre avis (en anglais)")

# Initialisation session state
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# Function to load example
def load_example(text):
    st.session_state.review_text = text

# Text area
user_input = st.text_area(
    "Enter your review",
    key="review_text",
    placeholder="Ex: This product is absolutely amazing! Best purchase I ever made.",
    height=150
)

# ── Examples ──────────────────────────────────────────────────
st.markdown("**Exemples à tester :**")

examples = [
    "This product is absolutely amazing! Best purchase I ever made.",
    "Terrible quality, broke after one day. Complete waste of money.",
    "It's okay, nothing special but does the job.",
    "I love it! Exceeded all my expectations, highly recommend.",
    "Very disappointing. Does not work as described."
]

cols = st.columns(2)

for i, ex in enumerate(examples):
    cols[i % 2].button(
        f"Exemple {i+1}",
        key=f"ex{i}",
        on_click=load_example,
        args=(ex,)
    )

# ── Predict ───────────────────────────────────────────────────
if st.button(" Analyser le sentiment", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning(" Veuillez entrer un avis avant d'analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            clean   = preprocess_text(user_input)
            vec     = vectorizer.transform([clean])
            pred    = model.predict(vec)[0]
            proba   = model.predict_proba(vec)[0]
            confidence = max(proba) * 100

        st.markdown("---")
        st.subheader(" Résultat")

        if pred == 1:
            st.success(f"## ✓ Avis POSITIF")
            st.balloons()
        else:
            st.error(f"## ✕ Avis NÉGATIF")

        # Confidence bar
        st.markdown(f"**Confiance du modèle : {confidence:.1f}%**")
        st.progress(confidence / 100)

        # Details
        col1, col2 = st.columns(2)
        col1.metric("Probabilité Négatif", f"{proba[0]*100:.1f}%")
        col2.metric("Probabilité Positif", f"{proba[1]*100:.1f}%")

        # Preprocessed text
        with st.expander(" Voir le texte après prétraitement"):
            st.code(clean)

st.markdown("---")
st.markdown("""
** À propos du modèle**
- **Algorithme** : Logistic Regression
- **Vectorisation** : TF-IDF (50,000 features)
- **Prétraitement** : Lowercase, suppression ponctuation, stop words, stemming
- **Dataset** : Amazon Reviews (3.6M train / 400K test)
- **Performance** : Accuracy 89.05% | F1-Score 89.10%
""")
