"""
This code automatically labels emotions on 3000 tweets using a pre-trained model,
namely j-hartmann/emotion-english-distilroberta-base from HuggingFace.
The code requires the transformers, torch and tqdm libraries.
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Configuration:

FICHIER_SOURCE  = "data/tweets_traites.csv"
FICHIER_SORTIE  = "data/tweets_entrainement.csv"
NB_TWEETS       = 10000
BATCH_SIZE      = 64    # tweets processed in parallel
RANDOM_STATE    = 42

# The model uses "joy"; we rename it "happiness" to match our labels
MAPPING_EMOTIONS = {"joy": "happiness"}

# 1: Loading and sampling the tweets to label.

print("Chargement du CSV source...")
df = pd.read_csv(FICHIER_SOURCE, sep=";", encoding="utf-8-sig")

# Remove empty tweets:
df = df.dropna(subset=["tweet_lemmatise"])
df = df[df["tweet_lemmatise"].str.strip() != ""]

# Random sample of 3000 tweets:
df_sample = df.sample(n=NB_TWEETS, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"{NB_TWEETS} tweets sélectionnés sur {len(df)} disponibles.")

# 2: Loading the HuggingFace emotion model.

print("\nChargement du modèle d'émotions (téléchargement au premier lancement)...")
classificateur = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,           # we only want the dominant emotion
    truncation=True,   # tweets longer than 512 tokens are truncated automatically
    max_length=512,
)
print("Modèle prêt.\n")


# 3: Labeling the tweets.

textes = df_sample["tweet_lemmatise"].tolist()
emotions = []

print(f"Étiquetage de {NB_TWEETS} tweets (batch_size={BATCH_SIZE})...")
for i in tqdm(range(0, NB_TWEETS, BATCH_SIZE)):
    batch = textes[i : i + BATCH_SIZE]
    resultats = classificateur(batch)
    for res in resultats:
        label = res[0]["label"]
        label = MAPPING_EMOTIONS.get(label, label)  # joy -> happiness
        emotions.append(label)


# 4: Creating the output file and exporting it as CSV.

df_sortie = pd.DataFrame({
    "tweet_filtre":   df_sample["tweet_lemmatise"].values,
    "pct_majuscules": df_sample["pct_majuscules"].values,
    "exclamations":   df_sample["exclamations"].values,
    "interrogations": df_sample["interrogations"].values,
    "emotion":        emotions,
})

df_sortie.to_csv(FICHIER_SORTIE, sep=";", index=False, encoding="utf-8-sig")

print(f"\nFichier exporté : {FICHIER_SORTIE} ({len(df_sortie)} tweets)")
print("\nRépartition des émotions détectées :")
print(df_sortie["emotion"].value_counts())