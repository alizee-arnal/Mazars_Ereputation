# Classification d'émotions dans des tweets COVID-19

Ce fichier permet de comprendre l'avancée de notre projet et comment utiliser les différents fichiers de code, notamment avec les bases de données. Le but principal du projet était de classifier les émotions de tweets relatifs au COVID-19 parmi : **anger, disgust, fear, happiness, neutral, sadness, surprise**.

## Structure du projet : 

Le projet est organisé en plusieurs branches Git, chacune correspondant à une étape du pipeline.

```
indexation                  → Chargement et nettoyage des données
feature/raphael-filtrage    → Prétraitement, filtrage, lemmatisation
étiquetage-automatique      → Étiquetage par lexique NRC + émojis
learning-modele-logistique  → Machine learning (régression logistique)
```

## Fichiers Python par branche

### Branche `indexation` / `feature/raphael-filtrage`

#### `indexation.py`
Chargement du dataset brut et premier nettoyage.
- Lit `data/covid19_tweets.csv` (~179 000 tweets)
- Extrait la colonne `text`
- Supprime les URLs via expression régulière
- Produit `liste_tweets_propres` (liste Python)
- **Bibliothèques** : `pandas`, `re`

#### `lemmatisation.py`
Prétraitement complet des tweets et extraction de features émotionnelles.
- **Filtrage stop-words** via spaCy (`en_core_web_sm`), avec conservation des négations (`not`, `never`, `no`...)
- **Répétitions consécutives** : mots répétés ≥ 2 fois d'affilée conservés (accentuation émotionnelle)
- **Émojis** conservés via la librairie `emoji`
- **Lemmatisation** avec exception pour les mots à lettres répétées (`WHATTTTT` → conservé tel quel)
- **Features numériques** calculées sur le texte original :
  - `pct_majuscules` : % de lettres majuscules
  - `exclamations` : nombre de `!`
  - `interrogations` : nombre de `?`
- Export vers `data/tweets_traites.csv`
- **Bibliothèques** : `spacy` (`en_core_web_sm`), `emoji`, `pandas`, `re`

#### `explore_data.py`
Exploration rapide du dataset : dimensions, colonnes, premières lignes.
- **Bibliothèques** : `pandas`

#### `general.py`
Utilitaires généraux partagés entre les scripts.
- **Bibliothèques** : `pandas`

---

### Branche `étiquetage-automatique`

#### `EI2.py`
Étiquetage des tweets par règles à base de lexique NRC et dictionnaire d'émojis.
- Charge le lexique NRC (10 catégories d'émotions)
- Détecte les négations pour inverser le sentiment
- Associe des émojis à chaque catégorie d'émotion (joie, tristesse, peur, colère...)
- Produit une étiquette par tweet selon le score dominant
- **Bibliothèques** : `pandas`, `nrclex`, `re`, `collections`

---

### Branche `learning-modele-logistique`

#### `etiquetage_auto.py`
Étiquetage automatique de tweets via un modèle pré-entraîné HuggingFace.
- Modèle : `j-hartmann/emotion-english-distilroberta-base`
- Échantillonne N tweets depuis `data/tweets_traites.csv` (paramétrable, 10 000 par défaut)
- Traitement par batch (batch_size=64) avec barre de progression
- Remapping `joy` → `happiness`
- Export vers `data/tweets_entrainement.csv`
- **Bibliothèques** : `pandas`, `transformers`, `torch`, `tqdm`

#### `machine_learning.py`
Pipeline principal de classification par régression logistique.
- **Entraînement** sur `data/tweets_entrainement.csv` (tweets auto-étiquetés)
- **Test** sur `data/Classeur2.xlsx` (297 tweets étiquetés manuellement)
- Features : TF-IDF (unigrammes + bigrammes, 5 000 max) + features numériques normalisées
- `class_weight="balanced"` pour compenser le déséquilibre des classes
- Affiche le rapport de classification et sauvegarde la matrice de confusion (`matrice_confusion.png`)
- Sauvegarde le modèle dans `modele_emotions.joblib`
- Résultats : accuracy ~42-53% selon le jeu de test
- **Bibliothèques** : `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `openpyxl`

#### `machine_learning_auto.py`
Variante de `machine_learning.py` entraînée sur l'étiquetage NRC (`bdd_etiquetee_emotions.csv`).
- Entraînement sur tous les tweets sauf les 50 000 derniers (réservés pour le test)
- Découpage positionnel (non aléatoire) pour une comparaison reproductible
- **Bibliothèques** : `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`

#### `evaluation_ml.py`
Évaluation comparative : ML vs étiquetage automatique par règles, mesurés contre la vérité terrain manuelle.
- Entraîne le modèle sur les tweets sans annotation manuelle
- Évalue sur les ~297 tweets annotés à la main
- Compare : **(A)** ML vs manuel, **(B)** règles auto vs manuel
- **Bibliothèques** : `pandas`, `numpy`, `scikit-learn`

#### `evaluation_new.py`
Même protocole qu'`evaluation_ml.py` appliqué à un nouvel étiquetage (`bdd_etiquetee_new.csv`).
- Récupère les features numériques depuis `bdd_etiquetee_emotions.csv` par alignement
- Produit : `fig_accuracy_new.png`, `fig_confusions_new.png`, `fig_distribution_new.png`
- Calcule le kappa de Cohen pour mesurer l'accord entre annotateurs
- **Bibliothèques** : `pandas`, `numpy`, `matplotlib`, `scikit-learn`

#### `rapport_figures.py`
Génère toutes les figures et métriques du rapport final.
- `fig_accuracy.png` : comparaison accuracy ML / règles auto / baseline
- `fig_confusions.png` : matrices de confusion côte à côte
- `fig_distribution.png` : distribution des classes
- Calcule le kappa de Cohen
- **Bibliothèques** : `pandas`, `numpy`, `matplotlib`, `scikit-learn`

---

## Installation

```bash
pip install pandas spacy emoji scikit-learn matplotlib joblib transformers torch tqdm openpyxl nrclex
python -m spacy download en_core_web_sm
```

## Ordre d'exécution

```bash
python indexation.py          # 1. Nettoyage des tweets
python lemmatisation.py       # 2. Filtrage + lemmatisation → tweets_traites.csv
python etiquetage_auto.py     # 3. Étiquetage automatique  → tweets_entrainement.csv
python machine_learning.py    # 4. Entraînement + évaluation
```

## Auteurs : Alizée Arnal, Ambre Chomel, Lina Khairouni, Yessine Jamal, Raphaël Kaufmann.

