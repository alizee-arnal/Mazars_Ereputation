"""
Ce code permet de génèrer les figures et métriques du rapport :
  - fig_accuracy.png        : comparaison des accuracy entre le ML, les règles auto et la baseline
  - fig_confusions.png      : matrices de confusion du ML et des règles 
  - fig_distribution.png    : distribution des classes (base auto vs annotation manuelle)
Il donne également toutes les métriques chiffrées, dont le kappa de Cohen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             cohen_kappa_score, ConfusionMatrixDisplay)

FICHIER = "bdd_etiquetee_emotions.csv"
SEP = ";"
TEXTE = "tweet_lemmatise"
NUM = ["pct_majuscules", "exclamations", "interrogations"]
AUTO = "emotion"
MANUEL = "Emotion"
RS = 42

# Chargement 
df = pd.read_csv(FICHIER, sep=SEP, encoding="utf-8-sig", low_memory=False)
df = df[[TEXTE] + NUM + [AUTO, MANUEL]]
df[TEXTE] = df[TEXTE].astype(str)

masque_ref = df[MANUEL].notna()
df_ref = df[masque_ref].copy()
df_train = df[~masque_ref].dropna(subset=[AUTO]).copy()

# Entraînement ML
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("texte", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000), TEXTE),
        ("num", StandardScaler(), NUM),
    ])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RS)),
])
pipe.fit(df_train[[TEXTE] + NUM], df_train[AUTO])

# Prédictions sur le jeu annoté 
y_vrai = df_ref[MANUEL]
y_ml = pipe.predict(df_ref[[TEXTE] + NUM])
y_auto = df_ref[AUTO]
labels = sorted(set(y_vrai) | set(y_ml) | set(y_auto))

# Métriques 
acc_ml = accuracy_score(y_vrai, y_ml)
acc_auto = accuracy_score(y_vrai, y_auto)
accord = accuracy_score(y_auto, y_ml)
kappa_ml = cohen_kappa_score(y_vrai, y_ml)
kappa_auto = cohen_kappa_score(y_vrai, y_auto)
base = y_vrai.value_counts(normalize=True).max()
classe_maj = y_vrai.value_counts().idxmax()

print("=== MÉTRIQUES ===")
print(f"n test (annotés)        : {len(df_ref)}")
print(f"n train                 : {len(df_train)}")
print(f"Baseline (toujours '{classe_maj}') : {base:.3f}")
print(f"Accuracy ML   vs manuel : {acc_ml:.3f}")
print(f"Accuracy auto vs manuel : {acc_auto:.3f}")
print(f"Kappa ML   vs manuel    : {kappa_ml:.3f}")
print(f"Kappa auto vs manuel    : {kappa_auto:.3f}")
print(f"Accord ML <-> auto      : {accord:.3f}")


# Génération de la figure 1 : barres d'accuracy

fig, ax = plt.subplots(figsize=(6, 4.2))
noms = [f"Baseline\n(toujours {classe_maj})", "Règles auto\nvs manuel", "ML\nvs manuel"]
vals = [base, acc_auto, acc_ml]
couleurs = ["#9e9e9e", "#ef9a4d", "#4d8eef"]
barres = ax.bar(noms, vals, color=couleurs)
for b, v in zip(barres, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("Accuracy contre la vérité terrain (annotation manuelle)")
plt.tight_layout()
plt.savefig("fig_accuracy.png", dpi=130)
plt.close()

# Génération de la figure 2 : matrices de confusion côte à côte

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, y_pred, titre in [(axes[0], y_ml, "ML vs manuel"),
                          (axes[1], y_auto, "Règles auto vs manuel")]:
    cm = confusion_matrix(y_vrai, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titre)
plt.suptitle("Matrices de confusion (lignes = vérité manuelle)")
plt.tight_layout()
plt.savefig("fig_confusions.png", dpi=130)
plt.close()


# Génération de la figure 3 : distribution des classes

fig, ax = plt.subplots(figsize=(8, 4.2))
ordre = sorted(set(df[AUTO].dropna()) | set(y_vrai))
auto_dist = df[AUTO].value_counts(normalize=True).reindex(ordre).fillna(0)
man_dist = y_vrai.value_counts(normalize=True).reindex(ordre).fillna(0)
x = np.arange(len(ordre))
w = 0.4
ax.bar(x - w / 2, auto_dist.values, w, label="Base auto (179k)", color="#ef9a4d")
ax.bar(x + w / 2, man_dist.values, w, label="Annotation manuelle (297)", color="#4d8eef")
ax.set_xticks(x)
ax.set_xticklabels(ordre, rotation=45, ha="right")
ax.set_ylabel("Proportion")
ax.set_title("Distribution des émotions : étiquetage auto vs annotation manuelle")
ax.legend()
plt.tight_layout()
plt.savefig("fig_distribution.png", dpi=130)
plt.close()

print("\nFigures générées : fig_accuracy.png, fig_confusions.png, fig_distribution.png")
