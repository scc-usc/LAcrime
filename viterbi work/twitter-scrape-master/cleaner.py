"""Author : Omkar Damle
	Date : 24th May 2017
	This is a testing code
	"""	
from __future__ import print_function
import settings
import tweepy
import dataset
from textblob import TextBlob
from sqlalchemy.exc import ProgrammingError
import json
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import string

tweet = raw_input('enter the tweet ->')

tweet = tweet.translate(None, string.punctuation)
relevant = False
print("Tweet after stripping punctuation : " + tweet)

print(type(tweet))
tweet_words = tweet.split(' ')

for word in tweet_words:
	stemmed_word = SnowballStemmer('english').stem(word)
	print(stemmed_word, end = "_")
	for track_word in settings.TRACK_TERMS:
		if stemmed_word == track_word:
			relevant = True    	            
			print("The first relevant term is : " + stemmed_word)
			break


if not relevant:
	print('The tweet is not relevant')
else:
	print('The tweet is relevant')

print("")




"""
-----------------------===========

		relevant = False
		tweet_words = status.text.split(' ')
#        print(status.text + "\n")

		for word in tweet_words:
			stemmed_word = SnowballStemmer('english').stem(word)
			for track_word in settings.TRACK_TERMS:
				if stemmed_word == track_word:
					relevant = True    	            
					print("The first relevant term is : " + stemmed_word)
					break


		if not relevant:
			return

		print(status.text + "\n")
"""