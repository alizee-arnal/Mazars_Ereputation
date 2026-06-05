#extraction des données
import pandas as pd

# 1. Charger le fichier CSV en mémoire (création d'un "DataFrame")
df = pd.read_csv('data/covid19_tweets.csv')

# 2. Extraire uniquement la colonne qui vous intéresse
colonne_tweets = df['text']

# 3. (Optionnel) Convertir cette colonne en une vraie liste Python 
# pour pouvoir boucler dessus plus tard lors du nettoyage
liste_tweets = colonne_tweets.tolist()

# Afficher les 5 premiers tweets pour vérifier que ça a marché
print(colonne_tweets.head())