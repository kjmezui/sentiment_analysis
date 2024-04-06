#%%
# Import the libraries
import pandas as pd 
from IPython import display

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#%%
# Loading the small_corpus dataset created in the "Creating dataset" milestone
reviews_dataset = pd.read_csv('/Users/jumper/transformers/amazon_reviews/small_corpus.csv')
reviews_dataset.head()

# %%
# Tokenize the dataset and  the words within
## 1- Using Treebankword Tokenizer
from nltk import TreebankWordTokenizer
from string import punctuation

reviews_dataset['rev_text_lower'] = reviews_dataset['reviewText'].apply(lambda review : str(review).translate(str.maketrans('','',punctuation)).replace("<br />", " ").lower())
reviews_dataset[['reviewText', 'rev_text_lower']].sample(2)

# %%
tb_tokenizer = TreebankWordTokenizer()
reviews_dataset['tb_tokens'] = reviews_dataset['rev_text_lower'].apply(lambda reviews : tb_tokenizer.tokenize(str(reviews)))
reviews_dataset[['reviewText', 'tb_tokens']].sample(3)

# %%
## 2- Using Casual Tokenizer
from nltk.tokenize.casual import casual_tokenize

reviews_dataset['casual_tokens'] = reviews_dataset['rev_text_lower'].apply(lambda reviews : casual_tokenize(reviews))
reviews_dataset[['reviewText', 'tb_tokens', 'casual_tokens']].sample(3)

# %%
# Words normalization
## Stemmatization
from nltk.stem.porter import PorterStemmer

stemmer  = PorterStemmer()
reviews_dataset['stem_tokens'] = reviews_dataset['tb_tokens'].apply(lambda words : [stemmer.stem(w) for w in words])
reviews_dataset[['tb_tokens', 'stem_tokens']].sample(2)

# %%
## Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn, sentiwordnet as swn

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    if tag.startswith('V'):
        return wn.VERB
    if tag.startswith('R'):
        return  wn.ADV
    if tag.startswith('N'):
        return wn.NOUN
    return None

lemmatizer  = WordNetLemmatizer()

def get_lemmas(tokens):
    lemmas = []
    for token in tokens:
        pos  = penn_to_wn(pos_tag([token])[0][1])
        if pos:
            lemma = lemmatizer.lemmatize(token, pos)
            if  lemma:
                lemmas.append(lemma)
    return lemmas

reviews_dataset['lemmas'] = reviews_dataset['tb_tokens'].apply(lambda tokens : get_lemmas(tokens))
reviews_dataset[['reviewText', 'stem_tokens', 'lemmas']].sample(3)

# %%
# Define the sentiment score predictor

def get_sentiment_score(tokens):
    score = 0
    tags = pos_tag(tokens)
    for word, tag in tags:
        wn_tag = penn_to_wn(tag)
        if not wn_tag:
            continue
        synsets  = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue
        # most common set:
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        score += (swn_synset.pos_score() - swn_synset.neg_score())

    return score  

# %%
# Test negative score
swn.senti_synset(wn.synsets("awful", wn.ADJ)[0].name()).neg_score()

# %%
# Test positive score
swn.senti_synset(wn.synsets("good", wn.ADJ)[0].name()).pos_score()

# %%
# Test the score using the get_sentiment_score methode
reviews_dataset['sentiment_score'] = reviews_dataset['lemmas'].apply(lambda tokens : get_sentiment_score(tokens))
reviews_dataset[['reviewText', 'lemmas', 'sentiment_score']].sample(5)
# %%
