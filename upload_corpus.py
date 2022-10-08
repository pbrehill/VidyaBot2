import openai
from openai.embeddings_utils import get_embedding
from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from itertools import chain

random.seed(11)


openai.api_key = "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"

tweets = pd.read_csv('corpus.csv')

limited_tweets = tweets.sample(n=3000)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

embeddings = []

def auto_embedding(query):
    result = get_embedding(query, engine='text-similarity-babbage-001')
    time.sleep(1)
    return result

embeddings = [auto_embedding(tweet) for tweet in limited_tweets.tweet]

limited_tweets['embeddings'] = embeddings

limited_tweets.to_pickle('embeddings1.pkl')

