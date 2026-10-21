"""
Emotion classification of COVID-19 tweets with logistic regression
 among the following emotions: disgust, anger, happiness, fear, sadness, neutral.
It uses several types of features:
  - the tweet text, vectorized with TF-IDF
  - precomputed numeric features (pct_majuscules, exclamations, interrogations)
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

# 1: Global configuration:

FICHIER_TRAIN  = "data/tweets_entrainement.csv"   # File containing the automatically labeled tweets.
FICHIER_TEST   = "data/Classeur2.xlsx"             # File containing the manually labeled tweets.
SEPARATEUR     = ";"

COLONNE_TEXTE = "tweet_filtre"
COLONNES_NUM  = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_LABEL = "emotion"

# Remapping of the emotions of Classeur2 to our labels:
REMAPPING_EMOTIONS = {
    "joy":           "happiness",
    "anticipation":  "neutral",
}

RANDOM_STATE = 42 # Seed used for reproducible results.

# 2: Loading the data.

def charger_train(chemin):
    '''Loads the training data from a CSV file.'''
    df = pd.read_csv(chemin, sep=SEPARATEUR)
    df = df.dropna(subset=[COLONNE_TEXTE, COLONNE_LABEL])
    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)
    print(f"Tweets d'entraînement : {len(df)}")
    print(df[COLONNE_LABEL].value_counts(), "\n")
    return df

def charger_test(chemin):
    '''Loads the test data from an Excel file.'''
    df = pd.read_excel(chemin)

    # Keep only the tweets with an emotion filled in
    df = df.dropna(subset=["text", "Emotion"])
    df = df.rename(columns={"text": COLONNE_TEXTE, "Emotion": COLONNE_LABEL})

    # Remapping of emotions to labels: joy -> happiness, anticipation -> neutral
    df[COLONNE_LABEL] = df[COLONNE_LABEL].replace(REMAPPING_EMOTIONS)

    # Computation of the numeric features from the text:
    df["pct_majuscules"] = df[COLONNE_TEXTE].apply(
        lambda t: round(sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1) * 100, 2)
    )
    df["exclamations"]   = df[COLONNE_TEXTE].apply(lambda t: t.count("!"))
    df["interrogations"] = df[COLONNE_TEXTE].apply(lambda t: t.count("?"))

    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

    print(f"Tweets de test (étiquetés à la main) : {len(df)}")
    print(df[COLONNE_LABEL].value_counts(), "\n")
    return df

# 3: Building the processing and classification pipeline.

def construire_pipeline():
    """
    This function essentially has 2 steps:
    The first, ColumnTransformer, applies a different treatment to each type of feature:
      - TF-IDF on the text
      - standardization (centering/scaling) on the numeric features (percentage of uppercase letters, number of exclamation/question marks)
    The second, Pipeline, combines these treatments with a logistic regression model.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("texte", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                min_df=2,             # ignore words that appear only once
                max_features=5000,    # limit the vocabulary size
            ), COLONNE_TEXTE),
            ("num", StandardScaler(), COLONNES_NUM),
        ]
    )

    modele = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Compensates for the class imbalance.
        C=1.0,                    # Hyperparameter corresponding to the inverse of lambda in RIDGE regression.
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", modele),
    ])

# 4: Training and evaluation of the model.

def entrainer_et_evaluer(df_train, df_test):
    ''' Trains the model on df_train and evaluates it on df_test, 
        while displaying a classification report and a confusion matrix.'''
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

# 5: Finding the most important words for each emotion.

def mots_importants(pipeline, n=10):
    """Displays, for each emotion, the words that push the most towards this class."""
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

# 6: Prediction on new tweets.

def predire(pipeline, df_nouveaux):
    """df_nouveaux must have the columns tweet_filtre and numeric features."""
    X = df_nouveaux[[COLONNE_TEXTE] + COLONNES_NUM]
    return pipeline.predict(X)

# 7: Main execution of the code:

if __name__ == "__main__":
    df_train = charger_train(FICHIER_TRAIN)
    df_test  = charger_test(FICHIER_TEST)
    pipeline = entrainer_et_evaluer(df_train, df_test)
    mots_importants(pipeline)

    # Save the trained model for reuse
    joblib.dump(pipeline, "modele_emotions.joblib")
    print("Modèle sauvegardé : modele_emotions.joblib")

    # To reload it later:
    #   pipeline = joblib.load("modele_emotions.joblib")
    #   predictions = predire(pipeline, df_a_classer)
    