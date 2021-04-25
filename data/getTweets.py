#!/usr/bin/env python

""" 
Sample code of getting tweet JSON objects by tweet ID lists.

You have to install tweepy (This script was tested with Python 2.6 and Tweepy 3.3.0)
https://github.com/tweepy/tweepy
and set its directory to your PYTHONPATH. 

You have to obtain an access tokens from dev.twitter.com with your Twitter account.
For more information, please follow:
https://dev.twitter.com/oauth/overview/application-owner-access-tokens

Once you get the tokens, please fill the tokens in the squotation marks in the
following "Access Information" part. For example, if your consumer key is 
LOVNhsAfB1zfPYnABCDE, you need to put it to Line 33
consumer_key = 'LOVNhsAfB1zfPYnABCDE' 



"""

# call user.lookup api to query a list of user ids.
import os

import tweepy
import sys
import json
import codecs
from tweepy.parsers import JSONParser

####### Access Information #################

# Parameter you need to specify
consumer_key = 'SAhpi9G7Dv10jRoChoh1ol7Ss'
consumer_secret = 'V4xfbOwLAh1SCjRZH8zfZlktJQKXFRAWxKQ0PQFCnWyYMCFwsK'
access_key = '1380144320095920131-lzfxPe7Njqb5aDBU5VUs2138pMxTTh'
access_secret = 's5Jop4hFgrvJ8UTobBnocCsAKwcmJV9Qu1V7QJIcypOcv'

inputFile = 'data.csv'
outputFile = 'tweet.json'

#############################################
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth_handler=auth, parser=JSONParser())

l = []
with open(inputFile, 'r') as inFile:
    with codecs.open(outputFile, 'w', encoding='utf8') as outFile:
        outFile.write("[")
        for line in inFile.readlines():
            l.append(line.split(",")[0].rstrip())
            if (len(l) >= 99):
                rst = api.statuses_lookup(id_=l)
                for tweet in rst:
                    outFile.write(json.dumps(tweet) + ",\n")
                l = []
        if (len(l) > 0):
            rst = api.statuses_lookup(id_=l)
            for tweet in rst:
                outFile.write(json.dumps(tweet) + ",\n")
        outFile.seek(-2, os.SEEK_CUR)
        outFile.write("] ")
