import tweepy
import json
import config
from pymongo import MongoClient

auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)

api = tweepy.API(auth)

# create stream listener
class StreamListener(tweepy.StreamListener):

	def __init__(self):
		_client = MongoClient()
		self.db = _client[config.db_name]
		self.tweets_coll = self.db[config.coll_name]
		self.fields = ['coordinates',
					   'created_at',
					   'entities',
					   'favorite_count',
					   'retweet_count',
					   'text',
					   'user']
		self.count = 0

	def on_data(self, data):
		try:
			tweet = json.loads(data)
			to_insert = {}
			for f in self.fields:
				to_insert[f] = tweet[f]
			to_insert['_id'] = tweet['id_str']
			self.tweets_coll.insert_one(to_insert)
			# with open('tweets.json', 'a') as f:
			# 	f.write(data)
			self.count += 1
			if self.count % 1000 == 0:
				print("Collected {0} tweets".format(self.count))
			return True
		except BaseException as e:
			print("Error on data: {}".format(e))
		return True


	def on_error(self, status_code):
		print(status_code)
		if status_code == 420:
			return False 
		else:
			return True
        		

# create a stream
myStream = tweepy.Stream(auth = api.auth, listener=StreamListener())

# start a stream 
myStream.sample(languages=['en'])
