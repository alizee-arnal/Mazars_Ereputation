import json
import re
from collections import Counter
from importlib import resources
import pandas as pd
import spacy


# Charger le lexique NRC
with resources.files("nrclex.data").joinpath("nrc_en.json").open(encoding="utf-8") as f:
    LEXIQUE = json.load(f)

# Les 10 catégories du NRC
EMOTIONS = ["disgust", "anger","anticipation", # chois des émotions et sentiments cibles, dans un ORDRE FIXE qui sert à départager les égalités
            "sadness", "surprise","joy","fear"]
SENTIMENTS = ["positive", "negative"]
CATEGORIES = EMOTIONS + SENTIMENTS

# enrichissement avec un vocabulaire spécifique au contexte COVID, lemmatisé pour correspondre à votre texte lemmatisé
covid_vocabularies_before_lemm = {#les mots neutres
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

    # les mots associés à la peur
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

    # les mots associés à la joie
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


# Lemmatisation du vocabulaire COVID pour correspondre à notre texte lemmatisé
# 1-Charger le modèle linguistique anglais de spaCy
nlp = spacy.load("en_core_web_sm")

def lemmatize_vocabulary(vocab_dict):
    lemmatized_dict = {}

    for category, terms in vocab_dict.items():
        lemmatized_terms = set()  # Utilisation d'un set pour éviter les doublons générés

        for term in terms:
            # Traiter l'expression (mot unique ou composé) avec spaCy
            doc = nlp(term)

            # Recomposer l'expression en utilisant le lemme de chaque mot
            # .lemma_.lower() assure que tout reste propre et en minuscules
            lemmatized_term = " ".join([token.lemma_.lower() for token in doc])

            lemmatized_terms.add(lemmatized_term)

        # Re-convertir en liste triée pour garder le format d'origine
        lemmatized_dict[category] = sorted(list(lemmatized_terms))

    return lemmatized_dict

# 2-Exécuter la lemmatisation
covid_vocabularies = lemmatize_vocabulary(covid_vocabularies_before_lemm)


# Definir les négations possibles
NEGATIONS = {"not", "no", "never", "without", "nor", "cannot","neither","couldn't","unable","don't"}
# Dictionnaire des émojis par catégorie d'émotion
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

#évaluer les sentiments dans chaque tweet
def compter(texte, nb_exclam=0):
    texte = str(texte)
    bruts = re.findall(r"[A-Za-z']+", texte)
    mots = [b.lower() for b in bruts]
    compte = Counter()

    # bonus d'intensité GLOBAL venant du tableau (les '!' ont été retirés du texte)
    bonus_exclam = min(nb_exclam, 3) * 0.5            # +0.5 par '!', plafonné à 3

    # emojis
    for emotion_fr, liste_emojis in emotions_emojis.items():
        cible = EMOJI_EMOTION[emotion_fr]
        for emoji in liste_emojis:
            nb = texte.count(emoji)
            if nb > 0:
                compte[cible] += 2 * nb

    # étiquetage par les mots du lexique + vocabulaire COVID
    for i, mot in enumerate(mots):
        poids = poids_intensite(bruts[i]) + bonus_exclam   # répétitions (texte) + '!' (tableau)
        cats = LEXIQUE.get(mot)
        if not cats: # pour le lexique spécifique au covid
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
        if precede_de_negation: # quand il y a une négation dans les 3 mots précédents, on inverse le poids des catégories positive/negative
            if "positive" in cats:
                compte["negative"] += 2 * poids
            if "negative" in cats:
                compte["positive"] += 2 * poids
        else:
            for c in cats: # les catégories du lexique sont comptées normalement
                compte[c] += poids
    return compte

# poids d'intensité : +1 pour les mots en MAJUSCULES, +0.5 pour les mots avec une lettre répétée (sooo, whatttt)
def poids_intensite(brut):
    """Poids >= 1 : lettres répétées (conservées dans le texte) + éventuelles majuscules."""
    poids = 1.0
    if len(brut) >= 3 and brut.isupper():          # mot crié en MAJUSCULES
        poids += 1.0
    if re.search(r"([a-zA-Z])\1\1", brut):         # lettres répétées : sooo, whattttt
        poids += 0.5
    return poids

## Déduction des émotions et sentiments
#  Émotion dominante
def emotion_dominante(compte):
    meilleure, score_max = "neutral", 0
    for emo in EMOTIONS:                                # parcours dans l'ordre fixe
        if compte.get(emo, 0) > score_max:            #  d'ou la liste ordonnée des émotions qui sert à départager les égalités
            score_max = compte[emo]
            meilleure = emo
    return meilleure


# Sentiment global
def sentiment_dominant(compte):
    pos, neg = compte.get("positive", 0), compte.get("negative", 0)
    if pos == 0 and neg == 0: # neutre si positif ninégatif
        return "neutral"
    return "positive" if pos >= neg else "negative"


# Appliquer à notre fichier de tweet néttoyés et lemmatisés
df = pd.read_csv("tweets_traites.csv", sep=";")

# on applique compter en lui donnant le texte et le nb d'exclamations de la même ligne
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
