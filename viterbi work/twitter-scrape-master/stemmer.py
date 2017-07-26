from __future__ import print_function
import settings
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
"""
stemmed = []
for term in settings.TRACK_TERMS:
	stemmed_term = SnowballStemmer("english").stem(term)
	print('\"' + stemmed_term + '\"',end = ', ')
	stemmed.append(stemmed_term)

"""
term = raw_input()
stemmed_term = SnowballStemmer("english").stem(term)
print('\"' + stemmed_term + '\"',end = ', ')
