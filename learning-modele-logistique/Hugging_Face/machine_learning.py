"""
Classification d'émotions de tweets COVID-19 par régression logistique
 parmis les émotions suivantes : disgust, anger, happiness, fear, sadness, neutral.
On le fait en ayant accès à plusieurs types de features :
  - le texte du tweet, vectorisé en TF-IDF
  - des features numériques déjà calculées (pct_majuscules, exclamations, interrogations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1 : Configuration globale :

FICHIER_TRAIN  = "data/tweets_entrainement.csv"   # Fichier correspondant aux tweets étiquetés automatiquement.
FICHIER_TEST   = "data/Classeur2.xlsx"             # Fichier correspondant aux tweets étiquetés à la main.
SEPARATEUR     = ";"

COLONNE_TEXTE = "tweet_filtre"
COLONNES_NUM  = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_LABEL = "emotion"

# Remapping des émotions du Classeur2 vers nos étiquettes :
REMAPPING_EMOTIONS = {
    "joy":           "happiness",
    "anticipation":  "neutral",
}

RANDOM_STATE = 42 # Facteur utilisé pour la reproductibilité des résultats.

# 2 : Chargement des données.

def charger_train(chemin):
    '''Charge les données d'entraînement depuis un fichier CSV.'''
    df = pd.read_csv(chemin, sep=SEPARATEUR)
    df = df.dropna(subset=[COLONNE_TEXTE, COLONNE_LABEL])
    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)
    print(f"Tweets d'entraînement : {len(df)}")
    print(df[COLONNE_LABEL].value_counts(), "\n")
    return df

def charger_test(chemin):
    '''Charge les données de test depuis un fichier Excel.'''
    df = pd.read_excel(chemin)

    # On garde uniquement les tweets avec une émotion renseignée
    df = df.dropna(subset=["text", "Emotion"])
    df = df.rename(columns={"text": COLONNE_TEXTE, "Emotion": COLONNE_LABEL})

    # Remapping des émotions vers des étiquettes : joy → happiness, anticipation → neutral
    df[COLONNE_LABEL] = df[COLONNE_LABEL].replace(REMAPPING_EMOTIONS)

    # Calcul des features numériques depuis le texte :
    df["pct_majuscules"] = df[COLONNE_TEXTE].apply(
        lambda t: round(sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1) * 100, 2)
    )
    df["exclamations"]   = df[COLONNE_TEXTE].apply(lambda t: t.count("!"))
    df["interrogations"] = df[COLONNE_TEXTE].apply(lambda t: t.count("?"))

    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

    print(f"Tweets de test (étiquetés à la main) : {len(df)}")
    print(df[COLONNE_LABEL].value_counts(), "\n")
    return df

# 3 : Construction du pipeline de traitement et de classification.

def construire_pipeline():
    """
    Cette fonction se divise essentiellement en 2 étapes :
    La première, ColumnTransformer, applique un traitement différent à chaque type de feature :
      - TF-IDF sur le texte
      - standardisation (centrage/réduction) sur les features numériques ( pourcentage de majuscule, nombre d'exclamations/interrogations )
    La seconde, Pipeline, combine ces traitements avec un modèle de regression logistique.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("texte", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrammes + bigrammes
                min_df=2,             # ignore les mots apparaissant 1 seule fois
                max_features=5000,    # limite la dimension du vocabulaire
            ), COLONNE_TEXTE),
            ("num", StandardScaler(), COLONNES_NUM),
        ]
    )

    modele = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Permet de compenser le déséquilibre des classes.
        C=1.0,                    # Hyperparamètre correspondant à l'inverse de lamba dans la regression RIDGE.
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", modele),
    ])

# 4 : Entraînement et évaluation du modèle.

def entrainer_et_evaluer(df_train, df_test):
    ''' Entraîne le modèle sur df_train et l'évalue sur df_test, 
        tout en  affichant un rapport de classification et une matrice de confusion.'''
    X_train = df_train[[COLONNE_TEXTE] + COLONNES_NUM]
    y_train = df_train[COLONNE_LABEL]

    X_test  = df_test[[COLONNE_TEXTE] + COLONNES_NUM]
    y_test  = df_test[COLONNE_LABEL]

    print(f"Train : {len(X_train)} tweets | Test : {len(X_test)} tweets\n")

    pipeline = construire_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION (test = labels manuels)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Matrice de confusion (test manuel)")
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=120)
    print("Matrice de confusion sauvegardée : matrice_confusion.png\n")

    return pipeline

# 5 : Détermination des mots les plus importants pour chaque émotion.

def mots_importants(pipeline, n=10):
    """Affiche, pour chaque émotion, les mots qui poussent le plus vers cette classe."""
    vectorizer = pipeline.named_steps["preprocessing"].named_transformers_["texte"]
    classifier = pipeline.named_steps["classifier"]
    noms_features = list(vectorizer.get_feature_names_out()) + COLONNES_NUM

    print("=" * 60)
    print("MOTS LES PLUS PRÉDICTIFS PAR ÉMOTION")
    print("=" * 60)
    for i, classe in enumerate(classifier.classes_):
        coefs = classifier.coef_[i]
        top = np.argsort(coefs)[-n:][::-1]
        mots = [noms_features[j] for j in top]
        print(f"\n{classe} : {', '.join(mots)}")
    print()

# 6 : Prédiction sur de nouveaux tweets.

def predire(pipeline, df_nouveaux):
    """df_nouveaux doit avoir les colonnes tweet_filtre et features numériques."""
    X = df_nouveaux[[COLONNE_TEXTE] + COLONNES_NUM]
    return pipeline.predict(X)

# 7 : Exécution principale du code : 

if __name__ == "__main__":
    df_train = charger_train(FICHIER_TRAIN)
    df_test  = charger_test(FICHIER_TEST)
    pipeline = entrainer_et_evaluer(df_train, df_test)
    mots_importants(pipeline)

    # Sauvegarde du modèle entraîné pour réutilisation
    joblib.dump(pipeline, "modele_emotions.joblib")
    print("Modèle sauvegardé : modele_emotions.joblib")

    # Pour recharger plus tard :
    #   pipeline = joblib.load("modele_emotions.joblib")
    #   predictions = predire(pipeline, df_a_classer)
    