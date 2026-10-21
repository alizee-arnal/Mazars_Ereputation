# -*- coding: utf-8 -*-
"""
Bag-of-words vectorization of the 1st column of the database.

Database: bdd_etiquetee_regles.csv
  - column 0: tweet_lemmatise (text to vectorize)
"""

import glob

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. Loading -----------------------------------------------------------
FICHIER = (glob.glob("../bdd_etiquet*csv") + glob.glob("bdd_etiquet*csv"))[0]
df = pd.read_csv(FICHIER, encoding="utf-8-sig")

TEXTE = df.columns[0]            # 1st column = tweet_lemmatise
textes = df[TEXTE].dropna().astype(str)
print(f"{len(textes)} tweets a vectoriser (colonne '{TEXTE}')")

# --- 2. Bag of words ----------------------------------------------------------
vectoriseur = CountVectorizer()         # occurrence counts (bag-of-words)
X = vectoriseur.fit_transform(textes)   # sparse matrix: tweets x words

vocab = vectoriseur.get_feature_names_out()
print(f"Matrice sac de mots : {X.shape[0]} tweets x {X.shape[1]} mots (creuse)")
print(f"Taille du vocabulaire : {len(vocab)} mots")
print("Exemple de mots :", list(vocab[:15]))
