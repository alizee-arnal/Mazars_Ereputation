"""
This code is similar to machine_learning.py, but it uses the bdd_etiquetee_reglesV1.csv database,
which contains tweets labeled automatically by rules ('emotion' column) and annotated manually ('Emotion' column).
It uses the same features as in machine_learning.py, namely:
- the tweet text, vectorized with TF-IDF
- precomputed numeric features (pct_majuscules, exclamations, interrogations)
To run the code, the pandas, scikit-learn,
 matplotlib and joblib libraries must be installed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# 1: Configuration of paths, column names...

# Database labeled automatically with NRC rules:
FICHIER_DONNEES = "data/bdd_etiquetee_reglesV1.csv"
SEPARATEUR = ";"  

COLONNE_TEXTE = "tweet_lemmatise"
COLONNES_NUM = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_LABEL = "emotion"

# We keep the last tweets to test our model:
N_TEST_FINAL = 50000

# We do not use the NRC columns (anger, disgust, fear...) as
# features. The 'emotion' column is derived from them, so including them would
# make the model relearn the rule that produced the label.

RANDOM_STATE = 42  # for reproducibility

# 2: Loading and preparing the data.

def charger_donnees(chemin):
    df = pd.read_csv(chemin, sep=SEPARATEUR, encoding="utf-8-sig")

    # We only keep the useful columns (the file contains
    # spurious 'Unnamed' columns coming from the export).
    df = df[[COLONNE_TEXTE] + COLONNES_NUM + [COLONNE_LABEL]]

    # Safety: remove the rows without text or without label
    df = df.dropna(subset=[COLONNE_TEXTE, COLONNE_LABEL])
    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

    print(f"Nombre de tweets étiquetés : {len(df)}")
    print("\nRépartition des classes :")
    print(df[COLONNE_LABEL].value_counts())
    print()

    return df

# 3: Building the Machine Learning pipeline.

def construire_pipeline():
    """
    ColumnTransformer = apply a different treatment to each type of feature:
      - TF-IDF on the text
      - standardization (centering/scaling) on the numeric features
    Then logistic regression on the concatenated set.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("texte", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams and bigrams
                min_df=2,             # ignore words that appear only once
                max_features=5000,    # limit the vocabulary size
            ), COLONNE_TEXTE),
            ("num", StandardScaler(), COLONNES_NUM),
        ]
    )

    modele = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Compensates for the class imbalance
        C=1.0,                    # Hyperparameter
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", modele),
    ])

# 4: Training and evaluation.

def entrainer_et_evaluer(df):
    '''Trains the model on the first N-50,000 tweets, then evaluates on the last 50,000.'''
    X = df[[COLONNE_TEXTE] + COLONNES_NUM]
    y = df[COLONNE_LABEL]

    n = len(df)
    if n <= N_TEST_FINAL:
        raise ValueError(
            f"Pas assez de tweets ({n}) pour réserver {N_TEST_FINAL} en test."
        )
    n_train = n - N_TEST_FINAL

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    print(f"Train : {len(X_train)} tweets | Test (50 000 derniers) : {len(X_test)} tweets\n")

    pipeline = construire_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation on the test set
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION (sur les 50 000 derniers tweets)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Matrice de confusion (étiquetage auto)")
    plt.tight_layout()
    plt.savefig("matrice_confusion_auto.png", dpi=120)
    print("Matrice de confusion sauvegardée : matrice_confusion_auto.png\n")

    return pipeline

# 5: Displaying the most important words for each class.

def mots_importants(pipeline, n=10):
    """Displays, for each emotion, the words that push the most towards this class.
    This is one of the big advantages of logistic regression: the coefficients can be read."""
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


# 6: Prediction function on new tweets.

def predire(pipeline, df_nouveaux):
    """df_nouveaux must have the columns tweet_lemmatise + numeric features."""
    X = df_nouveaux[[COLONNE_TEXTE] + COLONNES_NUM]
    return pipeline.predict(X)

# 7: Code execution.

if __name__ == "__main__":
    df = charger_donnees(FICHIER_DONNEES)
    pipeline = entrainer_et_evaluer(df)
    mots_importants(pipeline)

    # Save the trained model for reuse
    joblib.dump(pipeline, "modele_emotions_auto.joblib")
    print("Modèle sauvegardé : modele_emotions_auto.joblib")

