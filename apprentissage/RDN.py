# Réseau de neurone pour les émotions

import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. CHARGEMENT DES DONNÉES 

print("--- Étape 1 : Chargement de la base de données ---")


df = pd.read_csv("db.csv", encoding="utf-8-sig")

print(f"Données chargées : {len(df)} tweets")
print(f"Colonnes disponibles : {list(df.columns)}")

colonnes_scores = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise"]

#vérification que toutes les colonnes existent
for col in colonnes_scores:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante dans le CSV : '{col}'")

#definition des entrées (textes) et des cibles
tweets_textes = df['tweet_lemmatise'].astype(str).tolist()
y_scores_bruts = df[colonnes_scores].values
y_etiquettes_vraies = df['emotion'].astype(str).values

# Ajout d'une colonne "neutral" pour les tweets sans émotion
somme_autres_emotions = np.sum(y_scores_bruts, axis=1)
score_neutre = (somme_autres_emotions == 0).astype(int)
y_scores_bruts = np.column_stack((y_scores_bruts, score_neutre))
colonnes_scores_avec_neutral = colonnes_scores + ["neutral"]

# Normalisation des scores entre 0 et 1
max_score_possible = y_scores_bruts.max()
if max_score_possible == 0:
    raise ValueError("Tous les scores sont à 0, vérifiez votre fichier CSV.")
y_scores_normalises = y_scores_bruts / max_score_possible
print("Scores normalisés")

# Filtrage : on ne garde que les tweets dont l'étiquette 'emotion' est dans nos colonnes connues
etiquettes_valides = set(colonnes_scores_avec_neutral)
masque = np.array([e in etiquettes_valides for e in y_etiquettes_vraies])
tweets_textes = [t for t, m in zip(tweets_textes, masque) if m]
y_scores_normalises = y_scores_normalises[masque]
y_etiquettes_vraies = y_etiquettes_vraies[masque]
print(f"Tweets conservés après filtrage des étiquettes inconnues : {len(tweets_textes)}")

print("\nÉtape 2 : Extraction des embeddings avec SpaCy")
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])

vecteurs_tweets = []
for doc in nlp.pipe(tweets_textes, batch_size=64):  # batch_size réduit pour Colab
    vecteurs_tweets.append(doc.vector)

X = np.array(vecteurs_tweets)
print(f"Matrice d'embeddings X extraite. Taille : {X.shape}")

#Separation des données en 80% entrainement, 20% test

print("\nÉtape 3 : Séparation des données (80% Train / 20% Test)")

# Vérification qu'on a assez de données pour stratifier
from collections import Counter
comptage = Counter(y_etiquettes_vraies)
print(f"Distribution des émotions : {dict(comptage)}")

# On ne stratifie que si chaque classe a au moins 2 exemples
peut_stratifier = all(v >= 2 for v in comptage.values())

X_train, X_test, y_train_scores, y_test_scores, y_train_labels, y_test_labels = train_test_split(
    X, y_scores_normalises, y_etiquettes_vraies,
    test_size=0.2,
    random_state=42,
    stratify=y_etiquettes_vraies if peut_stratifier else None
)

print(f"Tweets pour l'entraînement : {X_train.shape[0]}")
print(f"Tweets pour le test        : {X_test.shape[0]}")

#ARCHITECTURE ET ENTRAÎNEMENT DU RÉSEAU DE NEURONES

print("\nÉtape 4 : Configuration et entraînement du Réseau de Neurones")

n_sorties = len(colonnes_scores_avec_neutral)  # 9(8 emotions+neutral)

model = models.Sequential([
    layers.Input(shape=(300,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(n_sorties, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train, y_train_scores,
    epochs=12,
    batch_size=32,          # batch réduit pour petits datasets
    validation_split=0.1,
    verbose=1
)
print("Entraînement terminé !")

#EVALUATION ET MATRICE DE CONFUSION

print("Étape 5 : Évaluation sur les données de Test")

y_pred_scores = model.predict(X_test)

indices_max_predits = np.argmax(y_pred_scores, axis=1)
y_pred_labels = np.array([colonnes_scores_avec_neutral[idx] for idx in indices_max_predits])

# On aligne les étiquettes présentes réellement pour le rapport
etiquettes_presentes = sorted(set(y_test_labels) | set(y_pred_labels))

print("\n=================== RAPPORT DE CLASSIFICATION ===================")
print(classification_report(y_test_labels, y_pred_labels, labels=etiquettes_presentes, zero_division=0))

print("====================== MATRICE DE CONFUSION ======================")
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=etiquettes_presentes)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquettes_presentes)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title("Matrice de Confusion (Émotion dominante prédite)")
plt.xlabel("Émotions Prédites")
plt.ylabel("Vraies Émotions")
plt.tight_layout()
plt.savefig("matrice_confusion_covid.png")
print("\nMatrice sauvegardée sous 'matrice_confusion_covid.png'.")
plt.show()
