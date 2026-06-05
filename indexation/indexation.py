#extraction des données
import pandas as pd
import re


# 1. Charger le fichier CSV en mémoire (création d'un "DataFrame")
df = pd.read_csv('data/covid19_tweets.csv')

# 2. Extraire uniquement la colonne qui vous intéresse
colonne_tweets = df['text']

# 3. (Optionnel) Convertir cette colonne en une vraie liste Python 
# pour pouvoir boucler dessus plus tard lors du nettoyage
liste_tweets = colonne_tweets.tolist()


# Imaginons que votre liste actuelle s'appelle 'liste_tweets'
liste_tweets_propres = []

# Le "motif" qui détecte tout ce qui commence par http ou https
motif_lien = r"https?://\S+"

for tweet in liste_tweets:
    # 1. On vérifie que le tweet est bien du texte (pour éviter les erreurs sur les cases vides)
    tweet = str(tweet)
    
    # 2. re.sub va remplacer le motif_lien par du vide ('') dans le tweet
    tweet_nettoye = re.sub(motif_lien, '', tweet)
    
    # 3. .strip() enlève les espaces vides inutiles au début ou à la fin
    liste_tweets_propres.append(tweet_nettoye.strip())

# Afficher les 5 premiers pour vérifier
print(liste_tweets_propres[:5])