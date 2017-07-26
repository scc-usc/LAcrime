"""Author : Omkar Damle
	Date : 24th May 2017
	This scraper is used to scrape only the geo tagged tweets in LA area
	"""	
from __future__ import print_function
import settings_geo
import tweepy
import dataset
from textblob import TextBlob
from sqlalchemy.exc import ProgrammingError
import json
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import string
from string import maketrans
import unicodedata

def startStream():
	count = 0
	while(True):
		print(count)
		count += 1
		try:
			stream.filter(locations=[-118.5,33.8,-118,34.2], languages=["en"])
		except Exception as e:
			print('Exception raised : ' + str(e))
			continue


db = dataset.connect(settings_geo.CONNECTION_STRING)        #????

class MyStreamListener(tweepy.StreamListener):

	def on_status(self, status):
		
#		print('-----------------------------------------------')

		if status.retweeted:
			return

		print('Original tweet : ' + status.text)

#ensures that only tweets with coordinates are stored
		if status.coordinates is None:
			return 

		description = status.user.description
		loc = status.user.location
		text = status.text
		coords = status.coordinates
		geo = status.geo
		name = status.user.screen_name
		user_created = status.user.created_at
		followers = status.user.followers_count
		id_str = status.id_str
		created = status.created_at
		retweets = status.retweet_count
		bg_color = status.user.profile_background_color
		blob = TextBlob(text)								#sentiment analysis
		sent = blob.sentiment

		if geo is not None:
			geo = json.dumps(geo)

		if coords is not None:
			coords = json.dumps(coords)

		table = db[settings_geo.TABLE_NAME]
		try:
			table.insert(dict(
				user_description=description,
				user_location=loc,
				coordinates=coords,
				text=text,
				geo=geo,
				user_name=name,
				user_created=user_created,
				user_followers=followers,
				id_str=id_str,
				created=created,
				retweet_count=retweets,
				user_bg_color=bg_color,
				polarity=sent.polarity,
				subjectivity=sent.subjectivity,
			))
		except ProgrammingError as err:
			print(err)

	def on_error(self, status_code):
		print(type(self))

		if status_code == 420:
			#returning False in on_data disconnects the stream
#			print("here")
			return False
#print("i am here")
auth = tweepy.OAuthHandler(settings_geo.TWITTER_APP_KEY, settings_geo.TWITTER_APP_SECRET)
auth.set_access_token(settings_geo.TWITTER_KEY, settings_geo.TWITTER_SECRET)
api = tweepy.API(auth)
"""
public_tweets = api.home_timeline()
for tweet in public_tweets:
	print tweet.text

print "now let us see the streams"
"""
stream_listener = MyStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
#stream.filter(track=settings.TRACK_TERMS, languages=["en"])

startStream()

