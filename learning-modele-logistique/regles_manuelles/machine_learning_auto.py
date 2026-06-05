"""
Ce code est similaire à machine_learning.py, mais il utilise la base bdd_etiquetee_reglesV1.csv, 
qui contient des tweets étiquetés automatiquement par règles (colonne 'emotion') et annotés manuellement (colonne 'Emotion').
On a accèss aux mêmes features que dans machine_learning.py, à savoir :
- le texte du tweet, vectorisé en TF-IDF
- des features numériques déjà calculées (pct_majuscules, exclamations, interrogations)
Pour pouvoir lancer le code, il est nécessaire d'avoir la bibliothèque pandas, scikit-learn,
 matplotlib et joblib installées.
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


# 1 : Configuration des chemins, noms de colonnes...

# Base étiquetée automatiquement par règles NRC :
FICHIER_DONNEES = "data/bdd_etiquetee_reglesV1.csv"
SEPARATEUR = ";"  

COLONNE_TEXTE = "tweet_lemmatise"
COLONNES_NUM = ["pct_majuscules", "exclamations", "interrogations"]
COLONNE_LABEL = "emotion"

# On garde les derniers tweets pour tester notre modèle : 
N_TEST_FINAL = 50000

# On n'utilise pas les colonnes NRC (anger, disgust, fear...) comme
# features. La colonne 'emotion' en est dérivée, donc les inclure férait
# que le modèle réapprendrait la règle qui a produit le label.

RANDOM_STATE = 42  # pour la reproductibilité

# 2 : Chargement et préparation des données.

def charger_donnees(chemin):
    df = pd.read_csv(chemin, sep=SEPARATEUR, encoding="utf-8-sig")

    # On ne garde que les colonnes utiles (le fichier contient des colonnes
    # parasites 'Unnamed' issues de l'export).
    df = df[[COLONNE_TEXTE] + COLONNES_NUM + [COLONNE_LABEL]]

    # Sécurité : on retire les lignes sans texte ou sans label
    df = df.dropna(subset=[COLONNE_TEXTE, COLONNE_LABEL])
    df[COLONNE_TEXTE] = df[COLONNE_TEXTE].astype(str)

    print(f"Nombre de tweets étiquetés : {len(df)}")
    print("\nRépartition des classes :")
    print(df[COLONNE_LABEL].value_counts())
    print()

    return df

# 3 : Construction du pipeline de Machine Learning.

def construire_pipeline():
    """
    ColumnTransformer = on applique un traitement différent à chaque type de feature :
      - TF-IDF sur le texte
      - standardisation (centrage/réduction) sur les features numériques
    Puis régression logistique sur l'ensemble concaténé.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("texte", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrammes et bigrammes
                min_df=2,             # ignore les mots apparaissant 1 seule fois
                max_features=5000,    # limite la dimension du vocabulaire
            ), COLONNE_TEXTE),
            ("num", StandardScaler(), COLONNES_NUM),
        ]
    )

    modele = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Permet de compenser le déséquilibre des classes
        C=1.0,                    # Hyperparamètre
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", modele),
    ])

# 4: Entraînement et évaluation.

def entrainer_et_evaluer(df):
    '''Entraîne le modèle sur les N-50 000 premiers tweets, puis évalue sur les 50 000 derniers.'''
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

    # Évaluation sur le test
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION (sur les 50 000 derniers tweets)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Matrice de confusion
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

# 5 : Affichage des mots les plus importants pour chaque classe.

def mots_importants(pipeline, n=10):
    """Affiche, pour chaque émotion, les mots qui poussent le plus vers cette classe.
    C'est l'un des gros avantages de la régression logistique : on peut lire les coefficients."""
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


# 6 : Fonction de prédiction sur de nouveaux tweets.

def predire(pipeline, df_nouveaux):
    """df_nouveaux doit avoir les colonnes tweet_lemmatise + features numériques."""
    X = df_nouveaux[[COLONNE_TEXTE] + COLONNES_NUM]
    return pipeline.predict(X)

# 7: Exécution du code.

if __name__ == "__main__":
    df = charger_donnees(FICHIER_DONNEES)
    pipeline = entrainer_et_evaluer(df)
    mots_importants(pipeline)

    # Sauvegarde du modèle entraîné pour réutilisation
    joblib.dump(pipeline, "modele_emotions_auto.joblib")
    print("Modèle sauvegardé : modele_emotions_auto.joblib")

