# Classification d'émotions dans des tweets COVID-19

Ce fichier permet de comprendre l'avancée de notre projet et comment utiliser les différents fichiers de code, notamment avec les bases de données. Le but principal du projet était de classifier les émotions de tweets relatifs au COVID-19 parmi : **anger, disgust, fear, happiness, neutral, sadness, surprise**.

## Structure du projet :

Le projet est organisé en **dossiers** à la racine, un par étape du pipeline.

- `filtrage/` : Chargement, nettoyage, filtrage et lemmatisation des tweets
- `etiquetage-automatique/` : Étiquetage par règles (lexique NRC + émojis + vocabulaire COVID)
- `learning-modele-logistique/` : Machine learning (régression logistique) et évaluation
- `apprentissage/` : Réseau de neurones (classification des émotions)
- `lina-index/` : Ancienne version du chargement / indexation (conservée pour archive)
- `data/` : Bases de données (fichiers CSV)

## Fichiers Python par dossier

### Dossier `filtrage/`
(regroupe le chargement, le nettoyage et la lemmatisation des tweets)

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
- **Répétitions consécutives** : les mots répétés plus de 2 fois d'affilée conservés 
- **Émojis** conservés via la librairie `emoji`
- **Lemmatisation** avec exception pour les mots à lettres répétées 
- **Features numériques** calculées sur le texte original :
  - `pct_majuscules` : % de lettres majuscules
  - `exclamations` : nombre de `!`
  - `interrogations` : nombre de `?`
- Export vers `data/tweets_traites.csv`
- **Bibliothèques** : `spacy` (`en_core_web_sm`), `emoji`, `pandas`, `re`

#### `explore_data.py`
Exploration rapide du dataset : dimensions, colonnes, premières lignes.
- **Bibliothèques** : `pandas`

### Dossier `etiquetage-automatique/`

#### `EI2.py`
Étiquetage des tweets par règles à base de lexique NRC et dictionnaire d'émojis.
- Charge le lexique NRC (10 catégories d'émotions)
- Détecte les négations pour inverser le sentiment
- Associe des émojis à chaque catégorie d'émotion (joie, tristesse, peur, colère...)
- Produit une étiquette par tweet selon le score dominant
- **Bibliothèques** : `pandas`, `nrclex`, `re`, `collections`

#### `etiquetage_automatique.py`
Version enrichie de l'étiquetage par règles (`EI2.py`), augmentée d'un vocabulaire spécifique au contexte COVID.
- Lexique NRC (émotions + valence positive/negative)
- **Vocabulaire COVID** (catégories `neutral`, `fear`, `joy`) lemmatisé avec spaCy (`en_core_web_sm`) pour correspondre au texte lemmatisé
- Dictionnaire d'émojis associés à chaque émotion (poids ×2)
- Gestion des négations (inversion de la valence) et poids d'intensité (MAJUSCULES, lettres répétées, `!`)
- Lit `tweets_traites.csv`, produit `bdd_etiquetee.csv` (une colonne par catégorie NRC + `emotion` + `sentiment`)
- **Bibliothèques** : `pandas`, `spacy` (`en_core_web_sm`), `nrclex`, `re`, `collections`, `json`

### Dossier `learning-modele-logistique/`

#### `etiquetage_auto.py`
Étiquetage automatique de tweets via un modèle pré-entraîné par HuggingFace.
- Modèle : `j-hartmann/emotion-english-distilroberta-base`
- Échantillonne N tweets depuis `data/tweets_traites.csv` (paramétrable, 10 000 par défaut)
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
Variante de `machine_learning.py` entraînée sur l'étiquetage NRC (`data/bdd_etiquetee_emotions.csv`).
- Entraînement sur tous les tweets sauf les 50 000 derniers (réservés pour le test)
- Découpage positionnel (non aléatoire) pour une comparaison reproductible
- **Bibliothèques** : `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`

#### `evaluation_ml.py`
Évaluation comparative : ML vs étiquetage automatique par règles, mesurés contre ce qui a été fait manuellement.
- Entraîne le modèle sur les tweets sans annotation manuelle
- Évalue sur les ~297 tweets annotés à la main
- Compare le modèle ML et l'étiquetage auto sur les tweets étiquetés manuellement.
- **Bibliothèques** : `pandas`, `numpy`, `scikit-learn`

#### `evaluation_new.py`
Même protocole qu'`evaluation_ml.py` appliqué à un nouvel étiquetage (`data/bdd_etiquetee_new.csv`).
- Récupère les features numériques depuis `data/bdd_etiquetee_emotions.csv` par alignement
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

### Dossier `apprentissage/`

#### `RDN.py`
Classification des émotions par réseau de neurones (alternative à la régression logistique).
- **Représentation** : embeddings spaCy `en_core_web_md` (vecteurs de dimension 300)
- **Cibles** : scores des 7 émotions + une classe `neutral` (tweets sans émotion), normalisés entre 0 et 1
- **Architecture (Keras / TensorFlow)** : Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(9, sigmoid)
- **Entraînement** : optimiseur `adam`, perte `binary_crossentropy`, 12 epochs, split 80% / 20% (stratifié si possible)
- **Évaluation** : rapport de classification + matrice de confusion (`matrice_confusion_covid.png`) sur l'émotion dominante (`argmax`)
- Lit `db.csv` (colonnes `tweet_lemmatise`, `emotion`, et scores `anger, anticipation, disgust, fear, joy, sadness, surprise`)
- **Bibliothèques** : `numpy`, `pandas`, `spacy` (`en_core_web_md`), `scikit-learn`, `tensorflow`, `matplotlib`

### Dossier `lina-index/`
Ancienne version des scripts de chargement et d'indexation (`indexation.py`, `explore_data.py`, `general.py`), conservée à titre d'archive. Remplacée par le dossier `filtrage/`.

---

## Installation

```bash
pip install pandas spacy emoji scikit-learn matplotlib joblib transformers torch tqdm openpyxl nrclex
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # nécessaire pour apprentissage/RDN.py
```

## Ordre d'exécution

> Remarque : les scripts utilisent des chemins relatifs vers `data/`. Exécute-les
> depuis la racine du projet (ou adapte les chemins) pour qu'ils trouvent les CSV.

```bash
python filtrage/indexation.py                          # 1. Nettoyage des tweets
python filtrage/lemmatisation.py                       # 2. Filtrage + lemmatisation → data/tweets_traites.csv
python learning-modele-logistique/etiquetage_auto.py   # 3. Étiquetage automatique  → data/tweets_entrainement.csv
python learning-modele-logistique/machine_learning.py  # 4. Entraînement + évaluation
```

## Auteurs : Alizée Arnal, Ambre Chomel, Lina Khairouni, Yessine Jamal, Raphaël Kaufmann.
