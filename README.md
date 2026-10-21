# Emotion classification of COVID-19 tweets

This file explains the progress of our project and how to use the different code files, in particular with the databases. The main goal of the project was to classify the emotions of COVID-19 tweets among: **anger, disgust, fear, happiness, neutral, sadness, surprise**.

## Project structure:

The project is organized into **folders** at the root, one per step of the pipeline.

- `filtrage/` (filtering): loading, cleaning, filtering and lemmatization of the tweets
- `etiquetage-automatique/` (automatic labeling): rule-based labeling (NRC lexicon + emojis + COVID vocabulary)
- `learning-modele-logistique/` (logistic model learning): machine learning (logistic regression) and evaluation
  - `Hugging_Face/`: labeling and model based on a pre-trained model
  - `regles_manuelles/` (manual rules): model and evaluation based on the rule-based labeling
- `RDN/` (neural network): neural network (emotion classification)
- `data/`: databases (CSV files)

## Python files by folder

### `filtrage/` folder
(groups the loading, cleaning and lemmatization of the tweets)

#### `indexation.py`
Loading of the raw dataset and first cleaning.
- Reads `covid19_tweets.csv` (~179,000 tweets)
- Extracts the `text` column
- Removes URLs with a regular expression
- Produces `liste_tweets_propres` (Python list)
- **Libraries**: `pandas`, `re`

#### `lemmatisation.py`
Full preprocessing of the tweets and extraction of emotional features.
- **Stop-word filtering** with spaCy (`en_core_web_sm`), keeping negations (`not`, `never`, `no`...)
- **Consecutive repetitions**: words repeated more than twice in a row are kept
- **Emojis** kept with the `emoji` library
- **Lemmatization** with an exception for words with repeated letters
- **Numeric features** computed on the original text:
  - `pct_majuscules`: % of uppercase letters
  - `exclamations`: number of `!`
  - `interrogations`: number of `?`
- Export to `tweets_traites.csv`
- **Libraries**: `spacy` (`en_core_web_sm`), `emoji`, `pandas`, `re`

#### `explore_data.py`
Quick exploration of the dataset: dimensions, columns, first rows.
- **Libraries**: `pandas`

### `etiquetage-automatique/` folder

#### `EI2.py`
Rule-based labeling of the tweets using the NRC lexicon and an emoji dictionary.
- Loads the NRC lexicon (10 emotion categories)
- Detects negations to invert the sentiment
- Associates emojis with each emotion category (joy, sadness, fear, anger...)
- Produces one label per tweet according to the dominant score
- **Libraries**: `pandas`, `nrclex`, `re`, `collections`

#### `etiquetage_automatique.py`
Enriched version of the rule-based labeling (`EI2.py`), extended with a vocabulary specific to the COVID context.
- NRC lexicon (emotions + positive/negative valence)
- **COVID vocabulary** (categories `neutral`, `fear`, `joy`) lemmatized with spaCy (`en_core_web_sm`) to match the lemmatized text
- Dictionary of emojis associated with each emotion (weight ×2)
- Negation handling (valence inversion) and intensity weights (CAPITAL LETTERS, repeated letters, `!`)
- Reads `tweets_traites.csv`, produces `bdd_etiquetee.csv` (one column per NRC category + `emotion` + `sentiment`)
- **Libraries**: `pandas`, `spacy` (`en_core_web_sm`), `nrclex`, `re`, `collections`, `json`

### `learning-modele-logistique/` folder

#### `Hugging_Face/etiquetage_auto.py`
Automatic labeling of tweets with a pre-trained HuggingFace model.
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Samples N tweets from `data/tweets_traites.csv` (configurable, 10,000 by default)
- Remapping `joy` → `happiness`
- Export to `data/tweets_entrainement.csv`
- **Libraries**: `pandas`, `transformers`, `torch`, `tqdm`

#### `Hugging_Face/machine_learning.py`
Main classification pipeline using logistic regression.
- **Training** on `data/tweets_entrainement.csv` (auto-labeled tweets)
- **Test** on a set of 297 manually labeled tweets
- Features: TF-IDF (unigrams + bigrams, 5,000 max) + normalized numeric features
- `class_weight="balanced"` to compensate for the class imbalance
- Displays the classification report and saves the confusion matrix (`matrice_confusion.png`)
- Saves the model to `modele_emotions.joblib`
- Results: accuracy ~42-53% depending on the test set
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `openpyxl`

#### `regles_manuelles/machine_learning_auto.py`
Variant of `machine_learning.py` trained on the rule-based labeling (`data/bdd_etiquetee_reglesV1.csv`).
- Trains on all tweets except the last 50,000 (kept for testing)
- Positional (non-random) split for a reproducible comparison
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`

#### `regles_manuelles/evaluation_ml.py`
Comparative evaluation: ML vs rule-based automatic labeling, both measured against the manual annotation.
- Trains the model on the tweets without manual annotation
- Evaluates on the ~297 hand-annotated tweets
- Compares the ML model and the automatic labeling on the manually labeled tweets
- **Libraries**: `pandas`, `numpy`, `scikit-learn`

#### `regles_manuelles/evaluation_new.py`
Same protocol as `evaluation_ml.py`, applied to the new labeling (`data/bdd_etiquetee_reglesV2.csv`).
- Retrieves the numeric features from `data/bdd_etiquetee_reglesV1.csv` by alignment
- Produces: `fig_accuracy_new.png`, `fig_confusions_new.png`, `fig_distribution_new.png`
- Computes Cohen's kappa to measure the agreement between annotators
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

#### `rapport_figures.py`
Generates all the figures and metrics of the final report.
- `fig_accuracy.png`: accuracy comparison ML / automatic rules / baseline
- `fig_confusions.png`: side-by-side confusion matrices
- `fig_distribution.png`: class distribution
- Computes Cohen's kappa
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

#### `ei_tweets/apprentissage.py`
Bag-of-words vectorization (`CountVectorizer`) of the first column of the database.
- **Libraries**: `pandas`, `scikit-learn`

### `RDN/` folder

#### `RDN.py`
Emotion classification with a neural network (alternative to logistic regression).
- **Representation**: spaCy `en_core_web_md` embeddings (300-dimensional vectors)
- **Targets**: scores of the 7 emotions + a `neutral` class (tweets without emotion), normalized between 0 and 1
- **Architecture (Keras / TensorFlow)**: Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(9, sigmoid)
- **Training**: `adam` optimizer, `binary_crossentropy` loss, 12 epochs, 80% / 20% split (stratified when possible)
- **Evaluation**: classification report + confusion matrix (`matrice_confusion_covid.png`) on the dominant emotion (`argmax`)
- Reads `db.csv` (columns `tweet_lemmatise`, `emotion`, and scores `anger, anticipation, disgust, fear, joy, sadness, surprise`)
- **Libraries**: `numpy`, `pandas`, `spacy` (`en_core_web_md`), `scikit-learn`, `tensorflow`, `matplotlib`

### `data/` folder
Project databases (CSV):
- `bdd_etiquetee_reglesV1.csv`: rule-based labeling (version 1) + manual annotation column
- `bdd_etiquetee_reglesV2.csv`: rule-based labeling (version 2, neutral by default)
- `bdd_score_reglesV1.csv`: detailed per-category scores of the rule-based labeling

---

## Installation

```bash
pip install pandas spacy emoji scikit-learn matplotlib joblib transformers torch tqdm openpyxl nrclex
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # required for RDN/RDN.py
```

## Execution order

> Note: the scripts use relative paths to `data/`. Run them
> from the project root (or adapt the paths) so that they find the CSV files.

```bash
python filtrage/indexation.py                                   # 1. Tweet cleaning
python filtrage/lemmatisation.py                                # 2. Filtering + lemmatization → tweets_traites.csv
python learning-modele-logistique/Hugging_Face/etiquetage_auto.py    # 3. Automatic labeling → data/tweets_entrainement.csv
python learning-modele-logistique/Hugging_Face/machine_learning.py   # 4. Training + evaluation
```

## Authors: Alizée Arnal, Ambre Chomel, Lina Khairouni, Yessine Jamal, Raphaël Kaufmann.
