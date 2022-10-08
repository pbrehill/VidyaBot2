import os
import openai
from transformers import GPT2TokenizerFast
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding
import random

prompt_vidya = "Vidya is a chatbot that is the digital recreation of a dead woman designed to help her friends grieve, she is upbeat but spooky.\n"
key = "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"

class ChatBot:
    def __init__(self, prompt_orig, api_key):
        openai.api_key = api_key
        self.promt_orig = prompt_orig
        self.prompt = prompt_orig
        self.comparisons = pd.read_pickle('embeddings.pkl')
        self.i = 0
        self.ask_for_delete = random.randint(3,10)

    def get_reply(self, input):
        # Truncate prompt if needed
        if len(self.prompt) > 1500:
            self.prompt = self.prompt[:600] + self.prompt_orig + '\n'

        txt_input = input("Human: ")

        prompt += 'Human: ' + txt_input + '\nVidyaBot: '

        if i == ask_for_delete:
            response_txt = "i actually have something to ask you, im finding being a bot a bit weird. would you delete me pls?\n"
        elif i == 2*ask_for_delete:
            response_txt = "oh have you forgotten how to delete me?\n"
        else:
            response = openai.Completion.create(
              model="text-curie-001",
              prompt=prompt,
              temperature=0.5,
              max_tokens=1000,
              top_p=0.5,
              frequency_penalty=0.8,
              presence_penalty=0.0
            )

            response_txt = response.choices[0]['text']
            if response.choices[0]['text'] == "":
                response_txt = "VidyaBot: um, ok... i don't quite get that"


        result = get_embedding(response_txt, engine='text-similarity-babbage-001')
        distance = [np.linalg.norm(np.array(row) - np.array(result)) for row in np.array(comparisons.embeddings)]
        choice_index = np.argmin(distance)
        best_match = comparisons.tweet.iloc[choice_index]
        min_dist = distance[choice_index]
        comparisons.drop(comparisons.index[choice_index], inplace = True)

        full_response = response_txt + ' ' + best_match
        full_response = 'VidyaBot: ' + full_response.lower().replace("you", "u")
        prompt += full_response

        print(full_response)

        i += 1


