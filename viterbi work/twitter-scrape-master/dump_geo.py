"""
Dumps db data into csv file
"""

import settings_geo
import tweepy
import dataset
from textblob import TextBlob

db = dataset.connect("sqlite:///tweets_LA_GEO_28may.db")
result = db["crime_data"].all()
dataset.freeze(result, format='csv', filename="tweets_LA_GEO_28may.csv")
