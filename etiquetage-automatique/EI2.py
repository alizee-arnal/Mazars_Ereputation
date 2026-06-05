import json
import re
from collections import Counter
from importlib import resources

import pandas as pd

# --- 1. Charger le lexique NRC (mot -> liste de catégories) ---
with resources.files("nrclex.data").joinpath("nrc_en.json").open(encoding="utf-8") as f:
    LEXIQUE = json.load(f)

# Les 10 catégories du NRC
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
SENTIMENTS = ["positive", "negative"]
CATEGORIES = EMOTIONS + SENTIMENTS

# Vos classes cibles, dans un ORDRE FIXE qui sert à départager les égalités
# (choix assumé et documenté : en cas d'égalité, la 1re de cette liste gagne)
CIBLES = ["fear", "anger", "sadness", "disgust", "surprise", "joy"]

# --- 2. Compter toutes les catégories d'un tweet ---
NEGATIONS = {"not", "no", "never", "without", "nor", "cannot","neither","couldn't","unable","don't"}
# Dictionnaire des émojis par catégorie d'émotion
emotions_emojis = {
    "joie": [
        "😂", "😀", "😁", "😃", "😄", "😊", "🥰", "😍", "🎉", "✨", 
        "🤩", "😸", "😹", "❤️", "💖", "💗", "💓", "☀️", "🕺", "💃", 
        "🥳", "🎈", "🍻", "🥂", "✌️", "👑", "👌", "👏", "💪"
    ],
    "tristesse": [
        "😭", "😢", "😞", "😓", "💔", "🙁", "🥺", "😩", "🥲", "🖤", 
        "😔", "😟", "😣", "😾", "🥀", "🌧️", "💧", "😰", "😿", "📉"
    ],
    "peur": [
        "😱", "😨", "😰", "😳", "💀", "👻", "😬", "🙀", "🫨", "🥶", 
        "⚠️", "🚨", "🏃‍♂️", "🏃‍♀️", "🫣", "🧿", "⛈️"
    ],
    "colère": [
        "😠", "😡", "🤬", "👿", "👿", "🖕", "😤", "💥", "👹", "👺", 
        "👊", "👎", "🔥", "💢", "⚔️", "🔪", "📢", "🚫"
    ],
    "anticipation": [
        "👀", "⏳", "⌛", "📅", "🚀", "🤔", "🔮", "🛎️", "🎯", "🔜", 
        "📈", "🤞", "✍️", "🗓️", "⏱️", "🎟️", "🎬"
    ],
    "dégout": [
        "🤮", "🤢", "💩", "🤢", "😒", "👎", "😷", "🙄", "🙅‍♂️", "🙅‍♀️", 
        "🤥", "🐀", "🪰", "🪳", "🚮"
    ],
    "neutre": [
        "😐", "😑", "😶", "💬", "👍", "👌", "📝", "🤷‍♂️", "🤷‍♀️", "🤖", 
        "💼", "💻", "🏠", "☕", "🚗", "📦", "📌", "🔍"
    ]
}


def compter(texte):
    bruts = re.findall(rf"[A-Za-z']+[!?]*", str(texte))
    mots = [re.sub(r"[!?]+$", "", b).lower() for b in bruts]    # versions propres
    compte = Counter()
    
    # --- 1. GESTION DES EMOJIS ---
    # On parcourt directement le texte d'origine pour y chercher les émojis
    for emotion, liste_emojis in emotions_emojis.items():
        for emoji in liste_emojis:
            # On compte combien de fois cet émoji apparaît dans le texte
            nb_apparitions = texte.count(emoji)
            if nb_apparitions > 0:
                compte[emotion] += nb_apparitions

    for i, mot in enumerate(mots):
        poids = poids_intensite(bruts[i])           # intensité ciblée sur ce mot
        cats = LEXIQUE.get(mot)  
        # None si mot absent
        if not cats:
            continue
        precede_de_negation = i > 0 and (mots[i-1] in NEGATIONS or mots[i-2] in NEGATIONS or mots[i-3] in NEGATIONS) 
        if precede_de_negation:
            # émotions niées -> non comptées ; valence inversée
            if "positive" in cats:
                compte["negative"] += poids
            if "negative" in cats:
                compte["positive"] += poids
        else:
            for c in cats:
                compte[c] += poids
    return compte



def poids_intensite(brut):
    """Poids >= 1 propre à CE mot (majuscules, '!', lettres répétées)."""
    poids = 1.0
    noyau = re.sub(r"[!?]+$", "", brut)
    if len(noyau) >= 3 and noyau.isupper():        # ce mot crié en MAJUSCULES
        poids += 1.0
    poids += min(brut.count("!"), 3) * 0.5         # '!' collés à ce mot
    if re.search(r"([a-zA-Z])\1\1", brut):         # lettres répétées (sooo)
        poids += 0.5
    return poids


# --- 3. Émotion dominante parmi vos classes (départage déterministe) ---
def emotion_dominante(compte):
    meilleure, score_max = "neutral", 0
    for emo in CIBLES:                                # parcours dans l'ordre fixe
        if compte.get(emo, 0) > score_max:            # '>' strict : 1er en cas d'égalité
            score_max = compte[emo]
            meilleure = emo
    return meilleure


# --- 4. Sentiment global (valence) ---
def sentiment_dominant(compte):
    pos, neg = compte.get("positive", 0), compte.get("negative", 0)
    if pos == 0 and neg == 0:
        return "neutral"
    return "positive" if pos >= neg else "negative"

    
# --- 5. Appliquer à toute la base ---
df = pd.read_csv("bdd_propre.csv")
comptes = df["text_clean"].apply(compter)

for cat in CATEGORIES:                                # une colonne par catégorie NRC
    df["nrc_" + cat] = comptes.apply(lambda c: c.get(cat, 0))

df["emotion_nrc"] = comptes.apply(emotion_dominante)
df["sentiment_nrc"] = comptes.apply(sentiment_dominant)

df.to_csv("bdd_etiquetee.csv", index=False, encoding="utf-8-sig")

print("=== Émotion dominante ===")
print(df["emotion_nrc"].value_counts(), "\n")
print("=== Sentiment global ===")
print(df["sentiment_nrc"].value_counts())