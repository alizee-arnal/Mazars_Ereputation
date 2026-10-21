
'''This code is used to: 
-Load the database, 
-Extract the content of the tweets as a list
-Clean the tweets by removing the links. '''
import pandas as pd
import re


# 1. Load the CSV file into memory (creation of a "DataFrame")
df = pd.read_csv('data/covid19_tweets.csv')

# 2. Extract only the column you are interested in
colonne_tweets = df['text']

# 3. (Optional) Convert this column into a real Python list
# so that we can loop over it later during the cleaning
liste_tweets = colonne_tweets.tolist()


# Let's say your current list is called 'liste_tweets'
liste_tweets_propres = []

# The "pattern" that detects everything starting with http or https
motif_lien = r"https?://\S+"

for tweet in liste_tweets:
    # 1. Check that the tweet is really text (to avoid errors on empty cells)
    tweet = str(tweet)
    
    # 2. re.sub replaces the motif_lien pattern with nothing ('') in the tweet
    tweet_nettoye = re.sub(motif_lien, '', tweet)
    
    # 3. .strip() removes useless whitespace at the start or end
    liste_tweets_propres.append(tweet_nettoye.strip())

# Display the first 5 to check
print(liste_tweets_propres[:5])