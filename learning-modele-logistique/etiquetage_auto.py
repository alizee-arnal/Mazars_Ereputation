"""
Le code permert l'étiquetage automatique d'émotions sur 3000 tweets via un modèle pré-entraîné,
qui est j-hartmann/emotion-english-distilroberta-base de HuggingFace.
Le code nécessite les bibliothèques transformers, torch et tqdm.
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Configuration : 

FICHIER_SOURCE  = "data/tweets_traites.csv"
FICHIER_SORTIE  = "data/tweets_entrainement.csv"
NB_TWEETS       = 10000
BATCH_SIZE      = 64    # tweets traités en parallèle 
RANDOM_STATE    = 42

# Le modèle utilise "joy" ; on le renomme "happiness" pour correspondre à nos étiquettes
MAPPING_EMOTIONS = {"joy": "happiness"}

# 1 : Chargement et échantillonnage des tweets à étiqueter.

print("Chargement du CSV source...")
df = pd.read_csv(FICHIER_SOURCE, sep=";", encoding="utf-8-sig")

# On retire les tweets vides :
df = df.dropna(subset=["tweet_lemmatise"])
df = df[df["tweet_lemmatise"].str.strip() != ""]

# Échantillon aléatoire de 3000 tweets :
df_sample = df.sample(n=NB_TWEETS, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"{NB_TWEETS} tweets sélectionnés sur {len(df)} disponibles.")

# 2: Chargement du modèle d'émotions de HuggingFace.

print("\nChargement du modèle d'émotions (téléchargement au premier lancement)...")
classificateur = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,           # on veut uniquement l'émotion dominante
    truncation=True,   # les tweets de plus de 512 tokens sont tronqués automatiquement
    max_length=512,
)
print("Modèle prêt.\n")


# 3 : Étiquetage des tweets.

textes = df_sample["tweet_lemmatise"].tolist()
emotions = []

print(f"Étiquetage de {NB_TWEETS} tweets (batch_size={BATCH_SIZE})...")
for i in tqdm(range(0, NB_TWEETS, BATCH_SIZE)):
    batch = textes[i : i + BATCH_SIZE]
    resultats = classificateur(batch)
    for res in resultats:
        label = res[0]["label"]
        label = MAPPING_EMOTIONS.get(label, label)  # joy → happiness
        emotions.append(label)


# 4 : Création du fichier de sortie et export en format CSV.

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