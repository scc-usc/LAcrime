"""
Author : Omkar Damle
Date : 30th May 2017

This document will take the tweets stored in the sql database and 
group them according to period of time desired. Then each period 
will have an associated document. The document will undergo preprocessing
 steps like removing url, puncutation, and stopwords, stemming. 
 Then we will make a document term matrix for each document. 
This will be the input to svd algorithm of skikit library. As an output,
we will get document feature matrix with reduced features. This will then
be used for prediction using regression like techniques.
"""

import dataset
import datetime
import settings_geo

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora

db = dataset.connect(settings_geo.CONNECTION_STRING)

table = db[settings_geo.TABLE_NAME]

rows = table.all()

#Time period of a document in minutes
docLength = 4*60

#list of documents
docs = []
tempDoc = ""
print(type(rows))

#this code is to get the time of the first tweet in the database
for row in rows:
	tempTime = row['created']
	break


#dateTimeFormat = '%Y-%m-%d %H:%M:%S.%f'
#print(type(tempTime))
#tempTime1 = datetime.datetime.strptime(tempTime,dateTimeFormat)
#print(tempTime1)

for row in rows:
	time = row['created']
#	time1 = datetime.datetime.strptime(time,dateTimeFormat)

	diff = time - tempTime
	diff1 = divmod(diff.total_seconds(),60)
	diffMin = diff1[0]
	tempDoc+=row['text']

	if diffMin > docLength:
		print(diff1[0])
		tempTime = time
		docs.append(tempDoc)
		tempDoc = ""


print(len(docs))

for i in range(11):
	print(len(docs[i]))



"""
Now we start with cleaning and preproccessing
"""

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in docs]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

print(type(dictionary))

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# doc2bow:::: converts a collection of words to its 
#bag-of-words representation: a list of (word_id, word_frequency) 2-tuples.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(type(doc_term_matrix))






