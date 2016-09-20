import json
import config
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from pymongo import MongoClient
from time import time

# connect to db
client = MongoClient()
db = client[config.db_name]
tweets_coll = db[config.coll_name]

# iterate over all tweets in collection
print("Cleaning {0} tweets".format(tweets_coll.count()))
t0 = time()
documents = []
cursor = tweets_coll.find({}, {'text': 1})
for document in cursor:
	text = document['text']
	# remove RT @*:, @*, #*, links 
	cleaned_tweet = ' '.join(re.sub(("(^RT)"
									 "|(@[A-Za-z0-9]+)"
									 "|(#[A-Za-z0-9]+)"
									 "|([^0-9A-Za-z \t])"
									 "|(\w+:\/\/\S+)"), " ", text).split())
	documents.append(cleaned_tweet)
print("Done in {:0.3f}s".format(time()-t0))

print("Transforming Corpora into tf-idf matrix")
t0 = time()
# set stopwords
stopset = set(stopwords.words('english'))
# tfidf vectorizer, on 1-3grams
vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1,3))
X = vectorizer.fit_transform(documents)
print("Done in {:0.3f}s".format(time()-t0))

print("Running LSA on tf-idf")
t0 = time()
lsa = TruncatedSVD(n_components=300) # TODO: Find correct n here-> appears to be large 
X_lsa = lsa.fit_transform(X)
print("Done in {:0.3f}s".format(time()-t0))

print("Variance explained by LSA components: {}%"
	.format(int(lsa.explained_variance_ratio_.sum() * 100)))

# terms = vectorizer.get_feature_names()
# for i, comp in enumerate(lsa.components_):
# 	termsInComp = zip(terms, comp)
# 	sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
# 	print("Concept {}".format(i))
# 	for term in sortedTerms:
# 		print(term[0])
# 	print()

# use elbow method to find good number of clusters
distortions = []
for i in range(1, 50):
	print(i)
	km = KMeans(n_clusters=i, init='k-means++', 
				n_init=10, max_iter=300, random_state=0)
	km.fit(X)
	distortions.append(km.inertia_)
plt.plot(range(1,50), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# k = 10
# km = KMeans(n_clusters=k, init='k-means++', n_init=1)

# print("Clustering sparse data with {}".format(km))
# t0 = time()
# km.fit(X_lsa)
# print("Done in {:0.3f}s".format(time()-t0))
# print()

# print("Distortion: {:0.2f}".format(km.inertia_))

print()

print("Top terms per cluster:")
original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()








