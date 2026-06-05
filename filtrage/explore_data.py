import pandas as pd

df = pd.read_csv("data/covid19_tweets.csv")

print("Dimensions :")
print(df.shape)

print("\nColonnes :")
print(df.columns)

print("\nPremières lignes :")
print(df.head())