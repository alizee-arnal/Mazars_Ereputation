import json
import re
from collections import Counter
from importlib import resources
import pandas as pd
import spacy


# Load the NRC lexicon
with resources.files("nrclex.data").joinpath("nrc_en.json").open(encoding="utf-8") as f:
    LEXIQUE = json.load(f)

# The 10 NRC categories
EMOTIONS = ["disgust", "anger","anticipation", # choice of target emotions and sentiments, in a FIXED ORDER used to break ties
            "sadness", "surprise","joy","fear"]
SENTIMENTS = ["positive", "negative"]
CATEGORIES = EMOTIONS + SENTIMENTS

# enrichment with a COVID-specific vocabulary, lemmatized to match your lemmatized text
covid_vocabularies_before_lemm = {#neutral words
    "neutral": [
        "coronavirus", "SARS-CoV-2", "epidemiology", "incubation period", "asymptomatic",
        "polymerase chain reaction", "antigen test", "antibodies", "viral load", "variant",
        "mutation", "transmission rate", "positivity rate", "efficacy rate", "clinical trial",
        "protocol", "quarantine", "isolation", "logistics", "remote work", "telecommuting",
        "hybrid model", "zoonotic spillover", "spike protein", "mRNA platform", "seroprevalence",
        "genomic sequencing", "lineage", "subvariant", "endemicity", "vector-based", "attenuated virus",
        "double-blind study", "placebo group", "peer-reviewed", "comorbidity factor", "aerosolization",
        "fomite transmission", "droplet spread", "contact tracing", "index case", "patient zero",
        "incidental finding", "triage protocol", "ct value", "false positive", "false negative",
        "epidemiological curve", "plateau phase", "excess mortality", "case fatality rate", "infection fatality rate",
        "statistical modeling", "digital green certificate", "qr code verification", "booster dose", "homologous boosting",
        "heterologous boosting", "subcutaneous injection", "intramuscular", "efficacy endpoint", "pharmacovigilance",
        "prophylaxis", "therapeutic agent", "monoclonal antibodies", "corticosteroids", "pulse oximeter",
        "saturation level", "pcr swab", "saliva test", "lateral flow assay", "sequencing data", "public health directive",
        "jurisdiction", "moratorium", "furlough scheme", "economic stimulus", "supply chain disruption",
        "nucleocapsid", "receptor-binding domain", "ace2 receptor", "viral shedding", "superspreading event",
        "recombinant strain", "seroconversion", "neutralization assay", "cross-reactivity", "t-cell response",
        "b-cell memory", "pathogenicity", "virulence", "r-naught", "effective reproduction number",
        "attack rate", "secondary attack rate", "pre-symptomatic", "paxlovid", "remdesivir",
        "molnupiravir", "intranasal vaccine", "multivalent vaccine", "cold chain logistics", "ultra-low temperature freezer",
        "covax facility", "vaccine passport", "trace-test-treat", "pcr pooling", "waste-water surveillance",
        "sentinel monitoring", "asymptomatic carrier", "community transmission", "local transmission", "imported case",
        "cluster mitigation", "containment strategy", "flattening the curve", "non-pharmaceutical interventions", "ventilatory support",
        "high-flow nasal cannula", "proning position", "baricitinib", "dexamethasone", "antiviral regimen",
        "asymptomatic screening", "rapid diagnostic test", "enzyme-linked immunosorbent assay", "herd immunity threshold", "pathogen genomics",

        "sanitizer", "hand sanitizer", "alcohol-based rub", "disinfectant wipe", "surface spray", "face shield", "surgical mask", "ffp2 respirator"
    ],

    # words associated with fear
    "fear": [
        "outbreak", "pandemic", "deadly", "fatal", "mortality", "death toll", "crisis",
        "emergency", "contamination", "contagious", "infection", "comorbidity", "respirator",
        "ventilator", "intensive care unit", "shortage", "lockdown", "curfew", "restrictions",
        "isolation", "loneliness", "anxiety", "superspreader", "infodemic", "panic buying",
        "wave", "surge", "exponential growth", "overwhelmed hospitals", "icu capacity", "oxygen scarcity",
        "collapse", "catastrophe", "threat", "lethal", "pathogenic", "cytokine storm", "acute respiratory distress syndrome",
        "hypoxia", "long covid", "brain fog", "chronic fatigue", "organ damage", "myocarditis",
        "underlying conditions", "high-risk category", "vulnerability", "containment breach", "hotspot", "red zone",
        "travel ban", "border closure", "mandatory confinement", "state of emergency", "martial law", "civil unrest",
        "anti-vax protests", "compliance fines", "police enforcement", "social alienation", "depression", "claustrophobia",
        "paranoia", "hypochondria", "doomscrolling", "cabin fever", "burnout", "caregiver fatigue",
        "bereavement", "unmarked graves", "morgue trucks", "supply hoarding", "toilet paper crisis", "black market",
        "counterfeit masks", "scam treatments", "misinformation", "conspiracy theory", "bioweapon", "leak",
        "vaccine hesitancy", "discrimination", "xenophobia", "scapegoating", "stigmatization", "economic ruin",
        "respiratory failure", "sepsis", "multi-organ failure", "thrombosis", "blood clots",
        "silent hypoxia", "post-acute sequelae", "dyspnea", "anosmia", "ageusia",
        "hyperinflammation", "intubation panic", "code red", "triage crisis", "medical rationing",
        "co-infection", "flurona", "black fungus", "mucormycosis", "morbidity",
        "lockdown fatigue", "social isolation", "zoom fatigue", "agoraphobia", "health anxiety",
        "coronaphobia", "stigmatization", "marginalization", "xenophobic attacks", "anti-maskers",
        "super-variants", "immune evasion", "vaccine resistance", "breakthrough infection", "waning immunity",
        "supply chain collapse", "food insecurity", "mass layoffs", "eviction crisis", "inflation spike",
        "economic recession", "business closures", "bankruptcy wave", "educational loss", "social distancing fines",
        "quarantine enforcement", "travel blacklists", "red-listing", "superspreading nightmare", "conspiracy echo-chamber",
        "misinformation wildfire", "deepfake science", "anti-science rhetoric", "medical distrust", "systemic failure",

        "price gouging", "sanitizer shortage", "hoarded stock", "chemical burns"
    ],

    # words associated with joy
    "joy": [
        "recovery", "survived", "discharge", "immunity", "vaccine breakthrough", "rollout",
        "protection", "antibodies", "reopening", "lifting restrictions", "end of lockdown", "reunion",
        "togetherness", "solidarity", "mutual aid", "essential workers", "applause", "clapping",
        "adaptation", "resilience", "hope", "progress", "normalization", "eradication", "suppression",
        "flattening the curve", "herd immunity", "immune response", "neutralizing antibodies", "scientific triumph",
        "unprecedented speed", "approval", "authorization", "distribution milestone", "jab", "vax",
        "fully vaccinated", "green pass", "travel resumption", "border reopening", "return to normalcy", "hugging again",
        "family reunion", "live music", "festivals", "outdoor dining", "vibrant streets", "economic recovery",
        "rebound", "job creation", "community spirit", "neighborhood support", "cheering", "balcony concerts",
        "healthcare heroes", "gratitude", "thankfulness", "optimism", "bright future", "silver lining",
        "innovation", "telehealth boom", "flexible working", "digital nomad", "work-life balance", "reduced emissions",
        "nature healing", "clearing skies", "pet adoption boom", "baking trend", "sourdough starter", "zoom party",
        "creative pivot", "triumph", "deliverance", "sanctuary", "well-being", "renewal",
        "remission", "convalescent success", "negative test", "viral clearance", "cellular immunity",
        "robust response", "life-saving drug", "vaccine efficacy", "emergency approval", "accelerated development",
        "global collaboration", "open-science", "genomic breakthrough", "vaxication", "shot in the arm",
        "double-jabbed", "booster protection", "freedom day", "unmasking", "reopening boom",
        "travel rebound", "tourism revival", "stadium cheering", "sold-out concerts", "theatre return",
        "hugging loved ones", "handshakes", "smiling faces", "public gatherings", "block parties",
        "community kindness", "charity drive", "volunteer surge", "care packages", "frontline applause",
        "scientific miracle", "historical achievement", "pandemic victory", "disease suppression", "declining cases",
        "empty icu beds", "hospital discharge cheer", "survivor stories", "post-pandemic boom", "market rally",
        "economic resurgence", "hybrid freedom", "wfh perks", "no commute", "family time",
        "urban greening", "wildlife returning", "cleaner air", "hobbies rediscovered", "mental health awareness",
        "collective resilience", "global relief", "bright horizon", "new beginnings", "celebration of life",

        "restocked shelves", "clean hands", "peace of mind"
    ]
}


# Lemmatize the COVID vocabulary to match our lemmatized text
# 1-Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

def lemmatize_vocabulary(vocab_dict):
    lemmatized_dict = {}

    for category, terms in vocab_dict.items():
        lemmatized_terms = set()  # Use a set to avoid generated duplicates

        for term in terms:
            # Process the expression (single or compound word) with spaCy
            doc = nlp(term)

            # Rebuild the expression using the lemma of each word
            # .lemma_.lower() keeps everything clean and lowercase
            lemmatized_term = " ".join([token.lemma_.lower() for token in doc])

            lemmatized_terms.add(lemmatized_term)

        # Convert back to a sorted list to keep the original format
        lemmatized_dict[category] = sorted(list(lemmatized_terms))

    return lemmatized_dict

# 2-Run the lemmatization
covid_vocabularies = lemmatize_vocabulary(covid_vocabularies_before_lemm)


# Define the possible negations
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
EMOJI_EMOTION = {"joie":"joy", "tristesse":"sadness", "peur":"fear",
                 "colère":"anger", "anticipation":"anticipation",
                 "dégout":"disgust", "neutre":"neutral"}

#evaluate the sentiments in each tweet
def compter(texte, nb_exclam=0):
    texte = str(texte)
    bruts = re.findall(r"[A-Za-z']+", texte)
    mots = [b.lower() for b in bruts]
    compte = Counter()

    # GLOBAL intensity bonus from the table column ('!' were removed from the text)
    bonus_exclam = min(nb_exclam, 3) * 0.5            # +0.5 per '!', capped at 3

    # emojis
    for emotion_fr, liste_emojis in emotions_emojis.items():
        cible = EMOJI_EMOTION[emotion_fr]
        for emoji in liste_emojis:
            nb = texte.count(emoji)
            if nb > 0:
                compte[cible] += 2 * nb

    # tagging with the lexicon words + COVID vocabulary
    for i, mot in enumerate(mots):
        poids = poids_intensite(bruts[i]) + bonus_exclam   # repetitions (text) + '!' (table)
        cats = LEXIQUE.get(mot)
        if not cats: # for the COVID-specific lexicon
            if mot in covid_vocabularies["neutral"]:
                compte["neutral"] += poids
            elif mot in covid_vocabularies["joy"]:
                compte["joy"] += poids
                compte["positive"] += poids
            elif mot in covid_vocabularies["fear"]:
                compte["fear"] += poids
                compte["sadness"] += poids
                compte["negative"] += poids
            continue
        precede_de_negation = any(mots[j] in NEGATIONS for j in range(max(0, i - 3), i))
        if precede_de_negation: # when there is a negation in the 3 preceding words, invert the weight of the positive/negative categories
            if "positive" in cats:
                compte["negative"] += 2 * poids
            if "negative" in cats:
                compte["positive"] += 2 * poids
        else:
            for c in cats: # the lexicon categories are counted normally
                compte[c] += poids
    return compte

# intensity weight: +1 for CAPITALIZED words, +0.5 for words with a repeated letter (sooo, whatttt)
def poids_intensite(brut):
    """Weight >= 1: repeated letters (kept in the text) + possible capitals."""
    poids = 1.0
    if len(brut) >= 3 and brut.isupper():          # word shouted in CAPITAL LETTERS
        poids += 1.0
    if re.search(r"([a-zA-Z])\1\1", brut):         # repeated letters: sooo, whattttt
        poids += 0.5
    return poids

## Deducing emotions and sentiments
#  Dominant emotion
def emotion_dominante(compte):
    meilleure, score_max = "neutral", 0
    for emo in EMOTIONS:                                # go through in the fixed order
        if compte.get(emo, 0) > score_max:            #  hence the ordered list of emotions used to break ties
            score_max = compte[emo]
            meilleure = emo
    return meilleure


# Overall sentiment
def sentiment_dominant(compte):
    pos, neg = compte.get("positive", 0), compte.get("negative", 0)
    if pos == 0 and neg == 0: # neutral if neither positive nor negative
        return "neutral"
    return "positive" if pos >= neg else "negative"


# Apply to our file of cleaned and lemmatized tweets
df = pd.read_csv("tweets_traites.csv", sep=";")

# apply compter giving it the text and the number of exclamations from the same row
comptes = df.apply(lambda row: compter(row["tweet_lemmatise"], row["exclamations"]), axis=1)

for cat in CATEGORIES:
    df[cat] = comptes.apply(lambda c, k=cat: c.get(k, 0))
df["emotion"]   = comptes.apply(emotion_dominante)
df["sentiment"] = comptes.apply(sentiment_dominant)
df.to_csv("bdd_etiquetee.csv", index=False, encoding="utf-8-sig")

print("=== Émotion dominante ===")
print(df["emotion"].value_counts(), "\n")
print("=== Sentiment global ===")
print(df["sentiment"].value_counts())
