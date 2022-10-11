import openai
from openai.embeddings_utils import get_embedding
from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from itertools import chain
import os
from tqdm import tqdm
from compress_pickle import compressed_pickle


random.seed(11)


openai.api_key = os.environ["OPENAI_API_KEY"]
# tweets = pd.read_csv('corpus.csv')

tweets=[]
with open("corpus.txt","r") as r:
    tweets=r.readlines()

limited_tweets = tweets

limited_tweets = pd.DataFrame(data = {'tweet': limited_tweets})

# already_processed_tweets = pd.read_pickl('embeddings.pkl')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

embeddings = []

def auto_embedding(query):
    result = get_embedding(query, engine='text-similarity-babbage-001')
    return result

for tweet in tqdm(limited_tweets.tweet):
    embeddings.append(auto_embedding(tweet))

limited_tweets['embeddings'] = embeddings

pickle = limited_tweets.to_pickle('embeddings1.pkl')

compressed_pickle('embeddings', pd.read_pickle('embeddings1.pkl'))


