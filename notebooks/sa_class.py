# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import nltk
from textblob_de import TextBlobDE as TextBlob
import numpy as np


# %%
class Text_Sentiment:
    def __init__(self, data):
        self.blob = TextBlob(str(data))
        self.n_sentences = self.blob.sentences 
        self.n_words = self.blob.tokens
        self.n_tags = self.blob.tags
        self.n_noun_phrases = self.blob.noun_phrases
        
    def text_sentiment(self):
        return self.blob.sentiment
    
    def sentence_sentiments(self):
        
        s_sentiments = []
        for sentence in self.n_sentences:
            s_sentiments.append(sentence.sentiment)
        return np.array(s_sentiments) 
        
    def agg_sentiment(self, fun=np.mean):
        return fun(self.sentence_sentiments(), axis=0)


# %%
import json
with open('article_example_1.txt') as json_file:
    data = json.load(json_file)


# %%
ts = Text_Sentiment(data['text'])


# %%
ts.text_sentiment()


# %%
np.array(ts.sentence_sentiments())


# %%
ts.agg_sentiment()

