"""
This code evaluates the new labeling (bdd_etiquetee_reglesV2.csv), with
the same indicators as evaluation_ml.py and rapport_figures.py.

The bdd_etiquetee_reglesV2.csv database only contains:
  [tweet_lemmatise, emotion (new rules), Emotions_manual]
The 3 numeric features are retrieved from bdd_etiquetee_reglesV1.csv

The indicators produced are:
  - accuracy & Cohen's kappa: ML vs manual and rules vs manual
  - ML agreement with the rules
  - detailed classification reports
  - figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             cohen_kappa_score, ConfusionMatrixDisplay)

FICHIER_NEW = "data/bdd_etiquetee_reglesV2.csv"   # new labeling and manual annotation
FICHIER_NUM = "data/bdd_etiquetee_reglesV1.csv"   # source of the 3 numeric features
SEP = ";"
TEXTE = "tweet_lemmatise"
NUM = ["pct_majuscules", "exclamations", "interrogations"]
AUTO = "emotion"             # new rules
MANUEL = "Emotions_manual"   # ground truth
RS = 42

# Loading and retrieving the numeric features by alignment
new = pd.read_csv(FICHIER_NEW, sep=SEP, encoding="utf-8-sig", low_memory=False)
num = pd.read_csv(FICHIER_NUM, sep=SEP, encoding="utf-8-sig", low_memory=False)
assert (new[TEXTE].astype(str).values == num[TEXTE].astype(str).values).all(), \
    "Les lignes des deux bases ne sont pas alignées !"

df = new[[TEXTE, AUTO, MANUEL]].copy()
for c in NUM:
    df[c] = num[c].values
df[TEXTE] = df[TEXTE].astype(str)

masque_ref = df[MANUEL].notna()
df_ref = df[masque_ref].copy()
df_train = df[~masque_ref].dropna(subset=[AUTO]).copy()

print(f"Tweets annotés à la main (test) : {len(df_ref)}")
print(f"Tweets d'entraînement           : {len(df_train)}")
print("\nNouvel étiquetage automatique (base complète) :")
print(df[AUTO].value_counts(), "\n")

# Identical pipeline
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("texte", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000), TEXTE),
        ("num", StandardScaler(), NUM),
    ])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RS)),
])
pipe.fit(df_train[[TEXTE] + NUM], df_train[AUTO])

# Predictions on the reference set
y_vrai = df_ref[MANUEL]
y_ml = pipe.predict(df_ref[[TEXTE] + NUM])
y_auto = df_ref[AUTO]
labels = sorted(set(y_vrai) | set(y_ml) | set(y_auto))

# Metrics
acc_ml = accuracy_score(y_vrai, y_ml)
acc_auto = accuracy_score(y_vrai, y_auto)
accord = accuracy_score(y_auto, y_ml)
kappa_ml = cohen_kappa_score(y_vrai, y_ml)
kappa_auto = cohen_kappa_score(y_vrai, y_auto)
base = y_vrai.value_counts(normalize=True).max()
classe_maj = y_vrai.value_counts().idxmax()


def rapport(titre, y_true, y_pred):
    print("=" * 64); print(titre); print("=" * 64)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


rapport("(A) MACHINE LEARNING vs annotation manuelle", y_vrai, y_ml)
rapport("(B) NOUVELLES RÈGLES vs annotation manuelle", y_vrai, y_auto)

print("=" * 64); print("SYNTHÈSE (nouvel étiquetage)"); print("=" * 64)
print(f"Baseline (toujours '{classe_maj}') : {base:.3f}")
print(f"Accuracy ML    vs manuel : {acc_ml:.3f}   | kappa : {kappa_ml:.3f}")
print(f"Accuracy règles vs manuel : {acc_auto:.3f}   | kappa : {kappa_auto:.3f}")
print(f"Accord ML <-> règles      : {accord:.3f}")

# Figures (same as rapport_figures.py, suffix _new):

# Fig 1: accuracy
fig, ax = plt.subplots(figsize=(6, 4.2))
noms = [f"Baseline\n(toujours {classe_maj})", "Règles\nvs manuel", "ML\nvs manuel"]
vals = [base, acc_auto, acc_ml]
for b, v in zip(ax.bar(noms, vals, color=["#9e9e9e", "#ef9a4d", "#4d8eef"]), vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("Accuracy vs vérité terrain (nouvel étiquetage)")
plt.tight_layout(); plt.savefig("fig_accuracy_new.png", dpi=130); plt.close()

# Fig 2: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, y_pred, titre in [(axes[0], y_ml, "ML vs manuel"),
                          (axes[1], y_auto, "Nouvelles règles vs manuel")]:
    cm = confusion_matrix(y_vrai, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titre)
plt.suptitle("Matrices de confusion -- nouvel étiquetage (lignes = vérité manuelle)")
plt.tight_layout(); plt.savefig("fig_confusions_new.png", dpi=130); plt.close()

# Fig 3: distribution
fig, ax = plt.subplots(figsize=(8, 4.2))
ordre = sorted(set(df[AUTO].dropna()) | set(y_vrai))
auto_dist = df[AUTO].value_counts(normalize=True).reindex(ordre).fillna(0)
man_dist = y_vrai.value_counts(normalize=True).reindex(ordre).fillna(0)
x = np.arange(len(ordre)); w = 0.4
ax.bar(x - w / 2, auto_dist.values, w, label="Nouvel étiquetage (179k)", color="#ef9a4d")
ax.bar(x + w / 2, man_dist.values, w, label="Annotation manuelle (297)", color="#4d8eef")
ax.set_xticks(x); ax.set_xticklabels(ordre, rotation=45, ha="right")
ax.set_ylabel("Proportion"); ax.legend()
ax.set_title("Distribution des émotions : nouvelles règles vs annotation manuelle")
plt.tight_layout(); plt.savefig("fig_distribution_new.png", dpi=130); plt.close()

print("\nFigures générées : fig_accuracy_new.png, fig_confusions_new.png, fig_distribution_new.png")
