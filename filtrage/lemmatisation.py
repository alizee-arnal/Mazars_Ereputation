''' 
This code cleans the database in several steps:
- computing the percentage of uppercase letters 
- counting emotional punctuation 
- filtering common words while keeping the important emotion markers (negation, repeated words, emojis)
- lemmatization (keeping the words with repeated letters)
It finally outputs a CSV file.
'''
import re
import spacy
import emoji
import pandas as pd
from indexation import liste_tweets_propres

# Load the English model (COVID-19 tweets are in English)
nlp = spacy.load("en_core_web_sm")

liste_vide = nlp.Defaults.stop_words

# Negation words to always keep (markers of anger, fear, sadness...)
NEGATIONS = {
    "not", "no", "nor", "never", "nothing", "nobody", "none",
    "neither", "nowhere", "n't", "cannot", "without", "most"
}

# Remove the negations from the stop-word list
liste_vide = liste_vide - NEGATIONS


def positions_repetition_consecutive(tokens, seuil=2):
    """Returns the indices of the tokens that are part of a consecutive repetition greater than 2"""
    positions = set()
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        if j - i >= seuil:
            positions.update(range(i, j))
        i = j
    return positions


def calculer_pourcentage_majuscules(tweets):
    """Returns the percentage of uppercase letters per tweet (on the original text)."""
    pourcentages = []
    for tweet in tweets:
        lettres = [c for c in tweet if c.isalpha()]
        if not lettres:
            pourcentages.append(0.0)
        else:
            nb_maj = sum(1 for c in lettres if c.isupper())
            pourcentages.append(round(nb_maj / len(lettres) * 100, 2))
    return pourcentages


def compter_ponctuation_emotion(tweets):
    """Returns the number of '!' and '?' per tweet (on the original text)."""
    resultats = []
    for tweet in tweets:
        nb_exclamation = tweet.count("!")
        nb_interrogation = tweet.count("?")
        resultats.append({
            "exclamations": nb_exclamation,
            "interrogations": nb_interrogation
        })
    return resultats


def filtrer_mots_courants(tweets, seuil_repetition=2):
    """
    Removes common words except:
    - negations
    - repeated words 
    - emojis 
    - Words with repeated letters are kept as they are.
    """
    tweets_filtres = []

    for tweet in tweets:
        # make_doc: tokenization only, without the ML pipeline (much faster)
        doc = nlp.make_doc(tweet.lower())
        tokens_alpha = [token.text for token in doc if token.is_alpha]

        indices_repetes = positions_repetition_consecutive(tokens_alpha, seuil_repetition)

        mots_gardes = []
        idx = 0
        for token in doc:
            # Case 1: emoji -> always keep
            if emoji.is_emoji(token.text):
                mots_gardes.append(token.text)
                continue

            # Case 2: non-alphabetic token (punctuation, digits...) -> ignore
            if not token.is_alpha:
                continue

            # Case 3: useful word or emphasis through consecutive repetition
            if (
                token.text not in liste_vide
                or idx in indices_repetes
            ):
                mots_gardes.append(token.text)
            idx += 1

        tweets_filtres.append(" ".join(mots_gardes))

    return tweets_filtres


def a_lettres_repetees(mot, seuil=3):
    """Returns True if the word contains a letter repeated >= seuil times in a row."""
    return bool(re.search(r'(.)\1{' + str(seuil - 1) + r',}', mot))


def lemmatiser(tweets_filtres):
    """
    Lemmatizes each word of the filtered tweet.
    Exception: words with repeated letters are kept as they are.
    Uses nlp.pipe() in batches for performance.
    """
    tweets_lemmatises = []

    for doc in nlp.pipe(tweets_filtres, disable=["ner", "parser"], batch_size=256):
        mots = []
        for token in doc:
            if a_lettres_repetees(token.text):
                mots.append(token.text)   # emotional emphasis -> no lemmatization
            else:
                mots.append(token.lemma_)
        tweets_lemmatises.append(" ".join(mots))

    return tweets_lemmatises

# Usage example

if __name__ == "__main__":
    tweets = liste_tweets_propres

    pourcentages_maj = calculer_pourcentage_majuscules(tweets)
    ponctuation = compter_ponctuation_emotion(tweets)
    tweets_filtres = filtrer_mots_courants(tweets)
    tweets_lemmatises = lemmatiser(tweets_filtres)

    df_resultats = pd.DataFrame({
        "tweet_lemmatise": [t.replace("\n", " ").replace("\r", " ") for t in tweets_lemmatises],
        "pct_majuscules":  pourcentages_maj,
        "exclamations":    [p["exclamations"]   for p in ponctuation],
        "interrogations":  [p["interrogations"] for p in ponctuation],
    })

    df_resultats.to_csv("data/tweets_traites.csv", index=False, encoding="utf-8-sig", sep=";")
    print(f"Fichier exporté : data/tweets_traites.csv ({len(df_resultats)} tweets)")
    print(df_resultats.head(3))
