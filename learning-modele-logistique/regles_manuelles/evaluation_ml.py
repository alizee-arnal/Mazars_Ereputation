"""
This code compares machine learning with the automatic rule-based labeling,
both measured against the manual annotation, on the bdd_etiquetee_reglesV1.csv database.

The bdd_etiquetee_reglesV1.csv file has the following columns:
  - 'tweet_lemmatise': text to vectorize
  - 'emotion' column: automatic labeling
  - 'Emotion' column: manual annotation
The number of manually labeled tweets is 297.

The evaluation principle is as follows:
  - the model is trained on all the tweets except the hand-annotated ones,
    using the automatic labels as the training target
  - it is then evaluated on the hand-annotated tweets.
This makes it possible to compare the performance of our Machine learning model
with the automatic rule-based model.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1: Configuration.

FICHIER_DONNEES = "data/bdd_etiquetee_reglesV1.csv"
SEPARATEUR = ";"

COLONNE_TEXTE = "tweet_lemmatise"
COLONNES_NUM = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_AUTO = "emotion"    # automatic labeling (training target)
COLONNE_MANUEL = "Emotion"  # manual annotation (ground truth)

RANDOM_STATE = 42

# 2: Loading and preparing the data.

df = pd.read_csv(FICHIER_DONNEES, sep=SEPARATEUR, encoding="utf-8-sig", low_memory=False)
df = df[[COLONNE_TEXTE] + COLONNES_NUM + [COLONNE_AUTO, COLONNE_MANUEL]]
df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

# Reference set, i.e. the rows annotated by hand
masque_ref = df[COLONNE_MANUEL].notna()
df_ref = df[masque_ref].copy()
# Training set, i.e. everything else, with a valid auto label
df_train = df[~masque_ref].dropna(subset=[COLONNE_AUTO]).copy()

print(f"Tweets annotés à la main (jeu de test)  : {len(df_ref)}")
print(f"Tweets d'entraînement (labels auto)     : {len(df_train)}")
print("\nRépartition des annotations manuelles :")
print(df_ref[COLONNE_MANUEL].value_counts(), "\n")

# 3: Machine Learning training (on automatic labels, excluding the test set).

def construire_pipeline():
    preprocessing = ColumnTransformer(
        transformers=[
            ("texte", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=5000,
            ), COLONNE_TEXTE),
            ("num", StandardScaler(), COLONNES_NUM),
        ]
    )
    modele = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", modele)])

X_train = df_train[[COLONNE_TEXTE] + COLONNES_NUM]
y_train = df_train[COLONNE_AUTO]

pipeline = construire_pipeline()
pipeline.fit(X_train, y_train)

# 4: Predictions on the reference set.

X_ref = df_ref[[COLONNE_TEXTE] + COLONNES_NUM]
y_vrai = df_ref[COLONNE_MANUEL]          # Manual
y_ml = pipeline.predict(X_ref)           # ML prediction
y_auto = df_ref[COLONNE_AUTO]            # automatic rule-based labeling

# Align the displayed labels on the union actually present
labels = sorted(set(y_vrai) | set(y_ml) | set(y_auto))

# 5: Evaluation and reports.

def rapport(titre, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print("=" * 64)
    print(titre)
    print("=" * 64)
    print(f"Accuracy : {acc:.3f}\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Matrice de confusion (lignes = vérité manuelle, colonnes = prédiction) :")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels), "\n")
    return acc

print("\nRéférence = annotation MANUELLE (vérité terrain)\n")
acc_ml = rapport("(A) MACHINE LEARNING  vs  annotation manuelle", y_vrai, y_ml)
acc_auto = rapport("(B) RÈGLES AUTO ('emotion')  vs  annotation manuelle", y_vrai, y_auto)

# ML agreement with the automatic rules:
accord = accuracy_score(y_auto, y_ml)

print("=" * 64)
print("SYNTHÈSE")
print("=" * 64)
print(f"Accuracy ML   vs manuel : {acc_ml:.3f}")
print(f"Accuracy auto vs manuel : {acc_auto:.3f}")
print(f"Écart (ML - auto)       : {acc_ml - acc_auto:+.3f}")
print(f"Accord ML avec auto     : {accord:.3f}")
