# -*- coding: utf-8 -*-
"""
Vectorisation sac de mots (bag-of-words) de la 1re colonne de la base.

Base : bdd_etiquetee_regles.csv
  - colonne 0 : tweet_lemmatise (texte a vectoriser)
"""

import glob

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. Chargement -----------------------------------------------------------
FICHIER = (glob.glob("../bdd_etiquet*csv") + glob.glob("bdd_etiquet*csv"))[0]
df = pd.read_csv(FICHIER, encoding="utf-8-sig")

TEXTE = df.columns[0]            # 1re colonne = tweet_lemmatise
textes = df[TEXTE].dropna().astype(str)
print(f"{len(textes)} tweets a vectoriser (colonne '{TEXTE}')")

# --- 2. Sac de mots ----------------------------------------------------------
vectoriseur = CountVectorizer()         # comptage d'occurrences (bag-of-words)
X = vectoriseur.fit_transform(textes)   # matrice creuse : tweets x mots

vocab = vectoriseur.get_feature_names_out()
print(f"Matrice sac de mots : {X.shape[0]} tweets x {X.shape[1]} mots (creuse)")
print(f"Taille du vocabulaire : {len(vocab)} mots")
print("Exemple de mots :", list(vocab[:15]))
