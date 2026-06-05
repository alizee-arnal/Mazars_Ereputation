#Liste vide (filtrage des mots courants)

import re
import spacy
import emoji
import pandas as pd
from indexation import liste_tweets_propres

# Charger le modèle anglais (tweets covid19 en anglais)
nlp = spacy.load("en_core_web_sm")

liste_vide = nlp.Defaults.stop_words

# Mots de négation à toujours conserver (marqueurs de colère, peur, tristesse...)
NEGATIONS = {
    "not", "no", "nor", "never", "nothing", "nobody", "none",
    "neither", "nowhere", "n't", "cannot", "without", "most"
}

# On retire les négations de la liste-vide
liste_vide = liste_vide - NEGATIONS


def positions_repetition_consecutive(tokens, seuil=2):
    """Retourne les indices des tokens faisant partie d'une répétition consécutive >= seuil."""
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
    """Retourne le pourcentage de lettres majuscules par tweet (sur le texte original)."""
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
    """Retourne le nombre de '!' et '?' par tweet (sur le texte original)."""
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
    Retire les mots courants sauf :
    - les négations (marqueurs d'émotion)
    - les mots répétés >= seuil_repetition fois d'affilée dans le tweet
    - les émojis (conservés tels quels comme marqueurs d'émotion)
    Les mots avec lettres répétées (ex. WHATTTTT) sont conservés tels quels.
    """
    tweets_filtres = []

    for tweet in tweets:
        # make_doc : tokenisation seule, sans pipeline ML (beaucoup plus rapide)
        doc = nlp.make_doc(tweet.lower())
        tokens_alpha = [token.text for token in doc if token.is_alpha]

        indices_repetes = positions_repetition_consecutive(tokens_alpha, seuil_repetition)

        mots_gardes = []
        idx = 0
        for token in doc:
            # Cas 1 : émoji → toujours conserver
            if emoji.is_emoji(token.text):
                mots_gardes.append(token.text)
                continue

            # Cas 2 : token non-alphabétique (ponctuation, chiffres...) → ignorer
            if not token.is_alpha:
                continue

            # Cas 3 : mot utile ou accentuation par répétition consécutive
            if (
                token.text not in liste_vide
                or idx in indices_repetes
            ):
                mots_gardes.append(token.text)
            idx += 1

        tweets_filtres.append(" ".join(mots_gardes))

    return tweets_filtres


def a_lettres_repetees(mot, seuil=3):
    """Retourne True si le mot contient une lettre répétée >= seuil fois d'affilée (ex. whattttt)."""
    return bool(re.search(r'(.)\1{' + str(seuil - 1) + r',}', mot))


def lemmatiser(tweets_filtres):
    """
    Lemmatise chaque mot du tweet filtré.
    Exception : les mots avec lettres répétées (emphase émotionnelle) sont conservés tels quels.
    Utilise nlp.pipe() en batch pour la performance.
    """
    tweets_lemmatises = []

    for doc in nlp.pipe(tweets_filtres, disable=["ner", "parser"], batch_size=256):
        mots = []
        for token in doc:
            if a_lettres_repetees(token.text):
                mots.append(token.text)   # emphase émotionnelle → pas de lemmatisation
            else:
                mots.append(token.lemma_)
        tweets_lemmatises.append(" ".join(mots))

    return tweets_lemmatises


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
