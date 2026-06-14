# ══════════════════════════════════════════════════════════════
#   PROJET DATA MINING — Analyse de Sentiments Amazon
#   Groupe 8 | Pr. KAISS Wijdane
# ══════════════════════════════════════════════════════════════



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import scipy.sparse as sp
import time
from tqdm import tqdm
import joblib

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
#
# print("Libraries imported!")









# ── Load Data ─────────────────────────────────────────────
print("\nLoading train.csv...")
df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train.csv")
print(f"Train loaded : {df_train.shape}")

print("\nLoading test.csv...")
df_test = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test.csv")
print(f"Test loaded  : {df_test.shape}")

# ── Basic Info ────────────────────────────────────────────────
print(f"\n{'='*50}")
print("TRAIN INFO:")
print(f"{'='*50}")
print(df_train.info())

print(f"\n{'='*50}")
print("FIRST 5 ROWS:")
print(f"{'='*50}")
print(df_train.head())

print(f"\n{'='*50}")
print("LABEL DISTRIBUTION:")
print(f"{'='*50}")
print(df_train['label'].value_counts())
print(df_train['label'].value_counts(normalize=True) * 100)













# ══════════════════════════════════════════════════════════════
# ANALYSE EXPLORATOIRE (EDA)
# ══════════════════════════════════════════════════════════════

# ── 1. Distribution des classes ───────────────────────────────
plt.figure(figsize=(6, 4))
counts = df_train['label'].value_counts().sort_index()
plt.bar(['Négatif (0)', 'Positif (1)'], counts.values,
        color=['#e74c3c', '#2ecc71'])
plt.title('Répartition des Sentiments')
plt.xlabel('Sentiment')
plt.ylabel("Nombre d'avis")
for i, v in enumerate(counts.values):
    plt.text(i, v + 10000, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_classes.png')
plt.show()
print(" Plot 1: Distribution des classes")

# ── 2. Longueur des avis (ALL DATA) ───────────────────────────
print("\n Computing review lengths on ALL data...")
df_train['text_length'] = df_train['text'].str.len()

plt.figure(figsize=(10, 4))
for label, color, name in [(0, '#e74c3c', 'Négatif'),
                            (1, '#2ecc71', 'Positif')]:
    subset = df_train[df_train['label'] == label]['text_length']
    plt.hist(subset, bins=50, alpha=0.6, color=color, label=name)

plt.title('Distribution de la Longueur des Avis')
plt.xlabel('Nombre de caractères')
plt.ylabel('Fréquence')
plt.legend()
plt.tight_layout()
plt.savefig('longueur_avis.png')
plt.show()

print(f"\n Longueur moyenne des avis (sur {len(df_train):,} avis):")
print(f"  Négatif : {df_train[df_train['label']==0]['text_length'].mean():.0f} caractères")
print(f"  Positif : {df_train[df_train['label']==1]['text_length'].mean():.0f} caractères")
print(f"  Minimum : {df_train['text_length'].min()} caractères")
print(f"  Maximum : {df_train['text_length'].max()} caractères")
print(f"  Médiane : {df_train['text_length'].median():.0f} caractères")

# ── 3. Exemples d'avis ────────────────────────────────────────
print(f"\n{'='*50}")
print(" EXEMPLES D'AVIS NÉGATIFS:")
print(f"{'='*50}")
for text in df_train[df_train['label']==0]['text'].head(3):
    print(f"→ {text[:150]}...")
    print()

print(f"{'='*50}")
print(" EXEMPLES D'AVIS POSITIFS:")
print(f"{'='*50}")
for text in df_train[df_train['label']==1]['text'].head(3):
    print(f"→ {text[:150]}...")
    print()











# ══════════════════════════════════════════════════════════════
# PRÉTRAITEMENT DU TEXTE
# ══════════════════════════════════════════════════════════════

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenization
    tokens = word_tokenize(text)
    # 4. Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    # 5. Stemming
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ── Test on one review first ──────────────────────────────────
print("=== TEST PREPROCESSING ===")
sample_text = df_train['text'].iloc[0]
print(f"BEFORE: {sample_text[:200]}")
print(f"AFTER : {preprocess_text(sample_text)[:200]}")

# ── Apply to ALL data ─────────────────────────────────────────

print(f"\n Preprocessing ALL train data ({len(df_train):,} rows)...")
print(" This will take time. Please wait...")
start = time.time()
df_train['clean_text'] = df_train['text'].apply(preprocess_text)
elapsed = time.time() - start
print(f" Train preprocessed in {elapsed/60:.1f} minutes!")

print(f"\n Preprocessing ALL test data ({len(df_test):,} rows)...")
start = time.time()
df_test['clean_text'] = df_test['text'].apply(preprocess_text)
elapsed = time.time() - start
print(f" Test preprocessed in {elapsed/60:.1f} minutes!")

# ── Save preprocessed data ────────────────────────────────────
# So we don't have to preprocess again next time!
print("\n Saving preprocessed data...")
df_train[['label', 'clean_text']].to_csv(
    r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv", index=False)
df_test[['label', 'clean_text']].to_csv(
    r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv", index=False)
print(" Saved!")

# ── Show result ───────────────────────────────────────────────
print("\n=== BEFORE vs AFTER ===")
for i in range(3):
    print(f"\n[{i+1}] BEFORE: {df_train['text'].iloc[i][:150]}")
    print(f"     AFTER : {df_train['clean_text'].iloc[i][:150]}")










# ══════════════════════════════════════════════════════════════
# VECTORISATION TF-IDF
# ══════════════════════════════════════════════════════════════

print("\nTF-IDF Vectorization...")
print("This take time on 3.6M rows...")

print("\nLoading preprocessed data...")
df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
df_test  = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv")
print(f"Train loaded : {df_train.shape}")
print(f"Test  loaded : {df_test.shape}")

# Fill any potential NaN values with empty strings
df_train['clean_text'] = df_train['clean_text'].fillna('')
df_test['clean_text'] = df_test['clean_text'].fillna('')

start = time.time()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=100000,   # keep top 100k words
    ngram_range=(1, 2),    # unigrams + bigrams
    min_df=5,              # word must appear in at least 5 docs
    max_df=0.95            # ignore words in more than 95% of docs
)

# Fit on train, transform both
X_train_tfidf = vectorizer.fit_transform(df_train['clean_text'])
X_test_tfidf  = vectorizer.transform(df_test['clean_text'])

y_train = df_train['label']
y_test  = df_test['label']

elapsed = time.time() - start
print(f"Vectorization done in {elapsed/60:.1f} minutes!")

print(f"\nTF-IDF Matrix:")
print(f"  Train shape : {X_train_tfidf.shape}")
print(f"  Test shape  : {X_test_tfidf.shape}")
print(f"  Vocabulary  : {len(vectorizer.vocabulary_):,} words")

# Show top words
print(f"\nSample vocabulary (first 20 words):")
vocab = sorted(vectorizer.vocabulary_.keys())
print(vocab[:20])

# Save TF-IDF matrix to disk
sp.save_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_train_tfidf.npz", X_train_tfidf)
sp.save_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_test_tfidf.npz",  X_test_tfidf)
print(" TF-IDF matrices saved!")










# # ── Load data
# print("Loading preprocessed data...")
# df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
# df_test  = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv")
# print(f"Train: {df_train.shape} | Test: {df_test.shape}")
#
# df_train['clean_text'] = df_train['clean_text'].fillna('')
# df_test['clean_text']  = df_test['clean_text'].fillna('')
#
# texts_train = df_train['clean_text'].tolist()
# texts_test  = df_test['clean_text'].tolist()
#
# # ── Step 1: FIT
# print("\nFitting TF-IDF on train data...")
# t0 = time.time()
#
# vectorizer = TfidfVectorizer(
#     max_features=50000,
#     ngram_range=(1, 1),
#     min_df=5,
#     max_df=0.95
# )
#
# vectorizer.fit(texts_train)
# print(f"Fit done in {(time.time()-t0)/60:.1f} min")
#
# # ── Step 2: TRANSFORM with progress bar
# def transform_in_batches(vectorizer, texts, batch_size=200_000, label=""):
#     batches = range(0, len(texts), batch_size)
#     results = []
#     t_start = time.time()
#
#     for i, start in enumerate(tqdm(batches, desc=label)):
#         batch = texts[start : start + batch_size]
#         results.append(vectorizer.transform(batch))
#
#         elapsed = time.time() - t_start
#         done, total = i + 1, len(batches)
#         eta = (elapsed / done) * (total - done)
#         tqdm.write(f"  Batch {done}/{total} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")
#
#     return sp.vstack(results)
#
# print("\nTransforming train...")
# X_train_tfidf = transform_in_batches(vectorizer, texts_train, label="Train")
#
# print("\nTransforming test...")
# X_test_tfidf = transform_in_batches(vectorizer, texts_test, label="Test")
#
# y_train = df_train['label']
# y_test  = df_test['label']
#
# # ── Save
# sp.save_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_train_tfidf.npz", X_train_tfidf)
# sp.save_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_test_tfidf.npz",  X_test_tfidf)
# print("\nTF-IDF matrices saved!")












# Load TF-IDF matrices
print(" Loading TF-IDF matrices...")
X_train_tfidf = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_train_tfidf.npz")
X_test_tfidf  = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_test_tfidf.npz")

# Load labels
df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
df_test  = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv")
y_train = df_train['label']
y_test  = df_test['label']

print(f" X_train : {X_train_tfidf.shape}")
print(f" X_test  : {X_test_tfidf.shape}")
print(f" y_train : {y_train.shape}")
print(f" y_test  : {y_test.shape}")









# ══════════════════════════════════════════════════════════════
# MODÉLISATION
# ══════════════════════════════════════════════════════════════

# ── Load saved TF-IDF matrices ────────────────────────────────
print("Loading TF-IDF matrices...")
X_train = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_train_tfidf.npz")
X_test  = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_test_tfidf.npz")

df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
df_test  = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv")
y_train = df_train['label']
y_test  = df_test['label']
print(f" Ready! X_train: {X_train.shape}")

def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")

    # Train
    print(f" Training...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f" Trained in {train_time:.1f}s")

    # Predict
    print(f" Predicting...")
    y_pred = model.predict(X_test)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n RESULTS:")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Négatif','Positif'])}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Négatif', 'Positif'])
    disp.plot(cmap='Blues')
    plt.title(f'Matrice de Confusion — {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_{name.replace(" ", "_")}.png')
    plt.show()

    return {'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'Train_time': f'{train_time:.1f}s'}

# ── Train 3 models ────────────────────────────────────────────
results = []

# 1. Naïve Bayes
results.append(train_and_evaluate(
    MultinomialNB(), "Naïve Bayes",
    X_train, y_train, X_test, y_test
))

# 2. Logistic Regression
results.append(train_and_evaluate(
    LogisticRegression(max_iter=1000, C=1.0),
    "Logistic Regression",
    X_train, y_train, X_test, y_test
))

# 3. Decision Tree
results.append(train_and_evaluate(
    DecisionTreeClassifier(max_depth=20, random_state=42),
    "Decision Tree",
    X_train, y_train, X_test, y_test
))

# ── Comparison Table ────────────────────────────────────
print(f"\n{'='*55}")
print(" COMPARISON")
print(f"{'='*55}")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ── Bar chart comparison ──────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, row in df_results.iterrows():
    ax.bar(x + i*width, [row[m] for m in metrics], width, label=row['Model'])

ax.set_ylabel('Score')
ax.set_title('Comparaison des Modèles — Analyse de Sentiments')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('comparaison_modeles.png')
plt.show()
print("\n All models trained and compared!")











# ── Load saved TF-IDF matrices ────────────────────────────────
print("Loading TF-IDF matrices...")
X_train = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_train_tfidf.npz")
X_test  = sp.load_npz(r"C:\Users\u1602\PycharmProjects\New Project\X_test_tfidf.npz")

df_train = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
df_test  = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\test_clean.csv")
y_train = df_train['label']
y_test  = df_test['label']
print(f"Ready! X_train: {X_train.shape}")

# ── Previous results (already computed) ──────────────────────
results = [
    {'Model': 'Naïve Bayes',         'Accuracy': 0.8353, 'Precision': 0.8390, 'Recall': 0.8299, 'F1': 0.8344, 'Train_time': '12.7s'},
    {'Model': 'Logistic Regression', 'Accuracy': 0.8905, 'Precision': 0.8871, 'Recall': 0.8949, 'F1': 0.8910, 'Train_time': '40.5s'},
]

# ── Train Decision Tree ───────────────────────────────────────
print("\n" + "="*55)
print("  MODEL: Decision Tree")
print("="*55)
print("Training...")
t0 = time.time()
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Trained in {train_time:.1f}s")

print("Predicting...")
y_pred_dt = dt_model.predict(X_test)

acc  = accuracy_score(y_test, y_pred_dt)
prec = precision_score(y_test, y_pred_dt)
rec  = recall_score(y_test, y_pred_dt)
f1   = f1_score(y_test, y_pred_dt)

print(f"\nRESULTS:")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred_dt, target_names=['Négatif','Positif'])}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Négatif', 'Positif'])
disp.plot(cmap='Blues')
plt.title('Matrice de Confusion — Decision Tree')
plt.tight_layout()
plt.savefig('confusion_Decision_Tree.png')
plt.show()

results.append({
    'Model': 'Decision Tree',
    'Accuracy': acc, 'Precision': prec,
    'Recall': rec, 'F1': f1,
    'Train_time': f'{train_time:.1f}s'
})




# ══════════════════════════════════════════════════════════════
# COMPARAISON
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print(" COMPARISON")
print("="*55)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ── Bar chart comparison ──────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.25
colors = ['#e74c3c', '#2ecc71', '#3498db']

fig, ax = plt.subplots(figsize=(10, 5))
for i, row in df_results.iterrows():
    ax.bar(x + i*width, [row[m] for m in metrics], width,
           label=row['Model'], color=colors[i])

ax.set_ylabel('Score')
ax.set_title('Comparaison des Modèles — Analyse de Sentiments Amazon')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0.7, 1.0)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig('comparaison_modeles.png')
plt.show()
print("Chart saved!")







# ══════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════

# Best model = Logistic Regression (89.05%)

print("\nRetraining Logistic Regression for saving...")
best_model = LogisticRegression(max_iter=1000, C=1.0)
best_model.fit(X_train, y_train)

joblib.dump(best_model, r"C:\Users\u1602\PycharmProjects\New Project\best_model.pkl")
print("Best model saved!")























# ══════════════════════════════════════════════════════════════
# TEST INTERACTIF
# ══════════════════════════════════════════════════════════════

# Load vectorizer
import pickle

# reload the vectorizer — rebuild it
print("\nReloading vectorizer...")
df_train_clean = pd.read_csv(r"C:\Users\u1602\PycharmProjects\New Project\train_clean.csv")
df_train_clean['clean_text'] = df_train_clean['clean_text'].fillna('')

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 1),
    min_df=5,
    max_df=0.95
)
vectorizer.fit(df_train_clean['clean_text'])
joblib.dump(vectorizer, r"C:\Users\u1602\PycharmProjects\New Project\vectorizer.pkl")
print("Vectorizer saved!")

# ── Preprocessing function ────────────────────────────────────
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ── Interactive test ──────────────────────────────────────────
print("\n" + "="*55)
print("  TEST INTERACTIF — Analyse de Sentiments")
print("="*55)

test_reviews = [
    "This product is absolutely amazing! Best purchase I ever made.",
    "Terrible quality, broke after one day. Complete waste of money.",
    "It's okay, nothing special but does the job.",
    "I love it! Exceeded all my expectations, highly recommend.",
    "Very disappointing. Does not work as described."
]

print("\nTesting on example reviews:")
for review in test_reviews:
    clean = preprocess_text(review)
    vec   = vectorizer.transform([clean])
    pred  = best_model.predict(vec)[0]
    proba = best_model.predict_proba(vec)[0]
    sentiment = "POSITIF " if pred == 1 else "NÉGATIF "
    confidence = max(proba) * 100
    print(f"\nAvis    : {review[:70]}...")
    print(f"Résultat: {sentiment} (confiance: {confidence:.1f}%)")

# ── Your own review ───────────────────────────────────────────
print("\n" + "="*55)
your_review = input("Entrez votre propre avis (en anglais) : ")
if your_review.strip():
    clean = preprocess_text(your_review)
    vec   = vectorizer.transform([clean])
    pred  = best_model.predict(vec)[0]
    proba = best_model.predict_proba(vec)[0]
    sentiment = "POSITIF " if pred == 1 else "NÉGATIF "
    confidence = max(proba) * 100
    print(f"\nRésultat: {sentiment}")
    print(f"Confiance: {confidence:.1f}%")

print("\n" + "="*55)
print("  PROJET TERMINÉ !")
print("="*55)
print(f"\nMeilleur modèle : Logistic Regression")
print(f"Accuracy        : 89.05%")
print(f"F1-Score        : 89.10%")
print(f"Dataset         : {len(df_train):,} avis Amazon")
print(f"\nFichiers sauvegardés:")
print(f"  best_model.pkl  ← modèle prêt à l'emploi")
print(f"  vectorizer.pkl  ← vectoriseur TF-IDF")
print(f"  comparaison_modeles.png")
print(f"  confusion_*.png")