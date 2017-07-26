"""
Author : Omkar Damle
Date : 25th May 2017
This python code is used to perform LDA on tweets 
reference : https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
"""

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


tweetFile='tweets_24may_night.dat'

with open(tweetFile) as f:
	content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
#print(len(content))

#for i in range(10):
#	print(content[i])

tweetSet = set(content)
tweets = list(tweetSet)

"""
Tweets contains the list of tweets!
"""
#print(len(tweets))

t1 = tweets[0:99]
t2 = tweets[99:199]
t3 = tweets[199:299]
t4 = tweets[299:399]
t5 = tweets[399:499]
t6 = tweets[499:599]

doc1 = ' '.join(t1)
doc2 = ' '.join(t2)
doc3 = ' '.join(t3)
doc4 = ' '.join(t4)
doc5 = ' '.join(t5)
doc6 = ' '.join(t6)

doc_complete = [doc1,doc2,doc3,doc4,doc5,doc6]

"""
Cleaning and preprocessing
"""

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
#??
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word.decode('utf-8')) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)


print(ldamodel.print_topics(num_topics=3, num_words=3))

