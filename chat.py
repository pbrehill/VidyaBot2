import os
import openai
from transformers import GPT2TokenizerFast
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding
import random
import re


class ChatBot:
    def __init__(self, prompt_orig, comparisons):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.prompt_orig = prompt_orig
        self.prompt = prompt_orig
        self.comparisons = comparisons
        self.i = 0
        self.ask_for_delete = random.randint(3,10)

    def get_reply(self, txt_input):
        # Truncate prompt if needed
        if len(self.prompt) > 1500:
            self.prompt = self.prompt[:600] + self.prompt_orig + '\n'

        self.prompt += 'Human: ' + txt_input + '\nVidyaBot: '

        if self.i == self.ask_for_delete:
            response_txt = "i actually have something to ask you, im finding being a bot a bit weird. would you delete me pls?\n"
        elif self.i == 2 * self.ask_for_delete:
            response_txt = "oh have you forgotten how to delete me?\n"
        else:
            response = openai.Completion.create(
              model="text-curie-001",
              prompt=self.prompt,
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
        distance = [np.linalg.norm(np.array(row) - np.array(result)) for row in np.array(self.comparisons.embeddings)]
        choice_index = np.argmin(distance)
        best_match = self.comparisons.tweet.iloc[choice_index]
        min_dist = distance[choice_index]
        self.comparisons.drop(self.comparisons.index[choice_index], inplace = True)

        full_response = response_txt + ' ' + best_match
        full_response = 'VidyaBot: ' + full_response.lower().replace("you", "u").replace("you", "u")

        if "human:" in full_response:
            full_response = full_response.split('human:', 1)[0]

        self.prompt += full_response

        self.i += 1

        return full_response


