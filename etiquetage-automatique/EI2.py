import json
import re
from collections import Counter
from importlib import resources

import pandas as pd

# --- 1. Load the NRC lexicon (word -> list of categories) ---
with resources.files("nrclex.data").joinpath("nrc_en.json").open(encoding="utf-8") as f:
    LEXIQUE = json.load(f)

# The 10 NRC categories
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
SENTIMENTS = ["positive", "negative"]
CATEGORIES = EMOTIONS + SENTIMENTS

# Your target classes, in a FIXED ORDER used to break ties
# (deliberate and documented choice: in case of a tie, the first one in this list wins)
CIBLES = ["fear", "anger", "sadness", "disgust", "surprise", "joy"]

# --- 2. Count all the categories of a tweet ---
NEGATIONS = {"not", "no", "never", "without", "nor", "cannot","neither","couldn't","unable","don't"}
# Dictionary of emojis per emotion category
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
    mots = [re.sub(r"[!?]+$", "", b).lower() for b in bruts]    # clean versions
    compte = Counter()
    
    # --- 1. EMOJI HANDLING ---
    # Go through the original text directly to look for emojis
    for emotion, liste_emojis in emotions_emojis.items():
        for emoji in liste_emojis:
            # Count how many times this emoji appears in the text
            nb_apparitions = texte.count(emoji)
            if nb_apparitions > 0:
                compte[emotion] += nb_apparitions

    for i, mot in enumerate(mots):
        poids = poids_intensite(bruts[i])           # intensity targeted at this word
        cats = LEXIQUE.get(mot)  
        # None if the word is absent
        if not cats:
            continue
        precede_de_negation = i > 0 and (mots[i-1] in NEGATIONS or mots[i-2] in NEGATIONS or mots[i-3] in NEGATIONS) 
        if precede_de_negation:
            # negated emotions -> not counted; valence inverted
            if "positive" in cats:
                compte["negative"] += poids
            if "negative" in cats:
                compte["positive"] += poids
        else:
            for c in cats:
                compte[c] += poids
    return compte



def poids_intensite(brut):
    """Weight >= 1 specific to THIS word (capitals, '!', repeated letters)."""
    poids = 1.0
    noyau = re.sub(r"[!?]+$", "", brut)
    if len(noyau) >= 3 and noyau.isupper():        # this word shouted in CAPITAL LETTERS
        poids += 1.0
    poids += min(brut.count("!"), 3) * 0.5         # '!' attached to this word
    if re.search(r"([a-zA-Z])\1\1", brut):         # repeated letters (sooo)
        poids += 0.5
    return poids


# --- 3. Dominant emotion among your classes (deterministic tie-breaking) ---
def emotion_dominante(compte):
    meilleure, score_max = "neutral", 0
    for emo in CIBLES:                                # go through in the fixed order
        if compte.get(emo, 0) > score_max:            # strict '>': the first one wins in case of a tie
            score_max = compte[emo]
            meilleure = emo
    return meilleure


# --- 4. Overall sentiment (valence) ---
def sentiment_dominant(compte):
    pos, neg = compte.get("positive", 0), compte.get("negative", 0)
    if pos == 0 and neg == 0:
        return "neutral"
    return "positive" if pos >= neg else "negative"

    
# --- 5. Apply to the whole dataset ---
df = pd.read_csv("bdd_propre.csv")
comptes = df["text_clean"].apply(compter)

for cat in CATEGORIES:                                # one column per NRC category
    df["nrc_" + cat] = comptes.apply(lambda c: c.get(cat, 0))

df["emotion_nrc"] = comptes.apply(emotion_dominante)
df["sentiment_nrc"] = comptes.apply(sentiment_dominant)

df.to_csv("bdd_etiquetee.csv", index=False, encoding="utf-8-sig")

print("=== Émotion dominante ===")
print(df["emotion_nrc"].value_counts(), "\n")
print("=== Sentiment global ===")
print(df["sentiment_nrc"].value_counts())