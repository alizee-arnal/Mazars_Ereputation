"""
Ce code permet de comparer le machine learning avec l'étiquetage automatique par règles,
mesurés tous les deux par rapport à l'annotation manuelle, sur la base bdd_etiquetee_reglesV1.csv.

Le fichier bdd_etiquetee_reglesV1.csv comporte les colonnes suivantes :
  - 'tweet_lemmatise' : texte à vectoriser
  - colonne 'emotion' : étiquetage automatique 
  - colonne 'Emotion' : annotation manuelle
Le nombre de tweets étiquetés manuellement est de 297 tweets.

Le principe de l'évaluation est le suivant :
  - le modèle est entraîné sur tous les tweets sauf ceux annotés à la main,
    en utilisant les labels automatiques comme cible d'entraînement
  - il est ensuite évalué sur les tweets annotés à la main.
Cela permet alors de comparer les perfomances de notre modèle de Machine learning 
avec le modèle de règles automatiques.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1 : Configuration.

FICHIER_DONNEES = "data/bdd_etiquetee_reglesV1.csv"
SEPARATEUR = ";"

COLONNE_TEXTE = "tweet_lemmatise"
COLONNES_NUM = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_AUTO = "emotion"    # étiquetage automatique (cible d'entraînement)
COLONNE_MANUEL = "Emotion"  # annotation manuelle (vérité terrain)

RANDOM_STATE = 42

# 2 : Chargement et préparation des données.

df = pd.read_csv(FICHIER_DONNEES, sep=SEPARATEUR, encoding="utf-8-sig", low_memory=False)
df = df[[COLONNE_TEXTE] + COLONNES_NUM + [COLONNE_AUTO, COLONNE_MANUEL]]
df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

# Jeu de référence, c'est-à-dire les lignes annotées à la main
masque_ref = df[COLONNE_MANUEL].notna()
df_ref = df[masque_ref].copy()
# Jeu d'entraînement, soit tout le reste, avec un label auto valide
df_train = df[~masque_ref].dropna(subset=[COLONNE_AUTO]).copy()

print(f"Tweets annotés à la main (jeu de test)  : {len(df_ref)}")
print(f"Tweets d'entraînement (labels auto)     : {len(df_train)}")
print("\nRépartition des annotations manuelles :")
print(df_ref[COLONNE_MANUEL].value_counts(), "\n")

# 3 : Entraînement du Machine Learning (sur labels automatiques, hors jeu de test).

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

# 4 : Prédictions sur le jeu de référence.

X_ref = df_ref[[COLONNE_TEXTE] + COLONNES_NUM]
y_vrai = df_ref[COLONNE_MANUEL]          # Manuel
y_ml = pipeline.predict(X_ref)           # prédiction du ML
y_auto = df_ref[COLONNE_AUTO]            # étiquetage automatique par règles

# On aligne les labels affichés sur l'union réellement présente
labels = sorted(set(y_vrai) | set(y_ml) | set(y_auto))

# 5: Évaluation et rapports.

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

# Accord ML avec les règles auto :
accord = accuracy_score(y_auto, y_ml)

print("=" * 64)
print("SYNTHÈSE")
print("=" * 64)
print(f"Accuracy ML   vs manuel : {acc_ml:.3f}")
print(f"Accuracy auto vs manuel : {acc_auto:.3f}")
print(f"Écart (ML - auto)       : {acc_ml - acc_auto:+.3f}")
print(f"Accord ML avec auto     : {accord:.3f}")
