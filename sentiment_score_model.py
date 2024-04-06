#%%
# Importing the libraries 
from nltk.tag import pos_tag 
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.stem import WordNetLemmatizer 
from string import punctuation
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
from IPython.display import display
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np

#%%
# 1 - Using sentiwornet scoring model
# Getting the wordnet tags
def get_wordnet_tags(tag):
    """
        Convert between the PennTreebank tags to simple Wordnet tags
    """    
    if tag.startswith("J"):
        return wn.ADJ
    elif tag.startswith("V"):
        return  wn.VERB
    elif tag.startswith("V"):
        return wn.NOUN,
    elif tag.startswith("R"):
        return wn.ADV
    return None

# Getting the sentiment score with sentiwordnet
def get_senti_score_of_reviews(text):
    """
        This method returns the sentiment score of a given text using SentiWordNet sentiment scores.
        input: text
        output: numeric (double) score, >0 means positive sentiment and <0 means negative sentiment.
    """     
    total_score = 0.0
    lemmatizer = WordNetLemmatizer()
    raw_sentence = sent_tokenize(text)

    for sentence in raw_sentence:
        senti_score = 0.0
        sentence = str(sentence).replace("<br />", " ").translate(str.maketrans('', '', punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(text)
        tags = pos_tag(tokens)
        for word, tag in tags:
            wn_tag = get_wordnet_tags(tag)
            if not wn_tag:
                continue           
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            #Tak the 1st, take the more common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            senti_score += swn_synset.pos_score() - swn_synset.neg_score()
        total_score += (senti_score / len(tokens))
    return (total_score + len(raw_sentence)) * 100

#%%
# Setting the screen dimensions for a better display
pd.set_option('display.max_column', None)
pd.set_option('display.max_colwidth', None)

# Loading the small corpus dataset creating in the creating dataset milestone
reviews = pd.read_csv('small_corpus.csv')
reviews.head()

#%%
# Removing NaN values from 'reviewText' column
reviews.dropna(subset=['reviewText'], inplace=True)
reviews.shape

# %%
# Defining the sentiwordnet scores from the 'reviewText'
reviews['swn_score'] = reviews['reviewText'].apply(lambda text : get_senti_score_of_reviews(text))
reviews[['reviewText', 'swn_score']].sample(2)

# %%
# Plotting the sentiwordnet scores
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
sns.histplot(x='swn_score', data=reviews.query('swn_score < 8 and swn_score > -8'), ax=ax)

# %%
# Defining the predicted sentiment data from the sentiwordnet scores
# And plotting the 'overall' grouped by the 'sentiwordnet sentiment'
reviews['swn_sentiment'] = reviews['swn_score'].apply(lambda x : "positive" if x>1 else ("negative" if x<0.5 else "neutral"))
reviews['swn_sentiment'].value_counts(dropna=False)

sns.countplot(x='overall', data=reviews, hue='swn_sentiment')

# %%
# Plotting the 'overall' against the 'sentiwordnet sentiment'
sns.boxenplot(x='overall', y='swn_sentiment', data=reviews)

# %%
# Re-plotting the 'overall' against 'sentiwordnet scores' from the dataset
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
sns.boxenplot(x='overall', y='swn_score', data=reviews)

# %%
#  Define the true sentiment data from the 'overall'
reviews['true_sentiment'] = reviews['overall'].apply(lambda x:
                                                     "positive" if x>=4 else 
                                                     ("neutral" if x==3 else  "negative"))

y_pred, y_true = reviews['swn_sentiment'].tolist(), reviews['true_sentiment'].tolist()
len(y_pred), len(y_true)

# %%
# Labels comparison using the confusion matrix
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_true, y_pred)
#labels = [['True Neg', 'False Pos'], ['False Neg', 'True Pos']]
#labels = np.asarray(labels).reshape(2, 2)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
sns.heatmap(cf_matrix, fmt='d', ax=ax, square=True, annot=True, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# %%
# Performance Assessment
## Positive prediction assessment prediction
tp, tn, fp, fn = 1310, 1088+265+70+188, 242+647, 142+547
recall = tp / (tp + fn) # Sensitivity (True Positive Rate)
specificity = tn / (tn + fp) # (True Negative Rate)
precision = tp / (tp + fp) # (Positive Predicted Value)
f1 = (2*tp) / (2*tp + fp + fn) # Accuracy
##  Note : f1-score  = 2* ((precision*recall) / (precision + recall))
print("recall : {}\nspecificity : {}\nprecision : {}\nf1 score : {}".format(recall, specificity, precision, f1))

# %%
## Negative prediction assessment prediction
tp, tn, fp, fn = 1088, 70+242+142+1310, 265+647, 188+547
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = (2*tp) / (2*tp + fp + fn)

print("recall : {}\nspecificity : {}\nprecision : {}\nf1 score : {}".format(recall, specificity, precision, f1))

# %%
# 2 - Using NLTK Opinion Lexicon scoring model
import nltk
from nltk.corpus import opinion_lexicon

nltk.download("opinion_lexicon")

# %%
# Definine the method to get the score using opinion_lexicon
pos_words = list(opinion_lexicon.positive())
neg_words  = list(opinion_lexicon.negative())

def get_sentiment_score_oplex(text):
    """ This method returns the sentiment score of a given text using nltk opinion lexicon.
        input: text
        output: numeric (double) score, >0 means positive sentiment and <0 means negative sentiment.
    """
    total_score = 0.0
    raw_sentences = sent_tokenize(text)
    for sentence in raw_sentences:
        sent_score = 0.0
        sentence = str(sentence).replace("<br />", " ").translate(str.maketrans('', '', punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(text)
        for token in tokens:
            sent_score =  sent_score +1 if token in pos_words else (sent_score -1 if 
                                                                     token in neg_words  else sent_score)
        total_score += (sent_score / len(tokens))
    
    return total_score  

#%%   
# Recall the dataset
reviews = pd.read_csv('/Users/jumper/transformers/amazon_reviews/small_corpus.csv')
reviews.dropna(subset=['reviewText'], inplace=True)

#%%
# Defining and plotting the oplex score 
reviews['oplex_score'] = reviews['reviewText'].apply(lambda text : get_sentiment_score_oplex(text))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
sns.histplot(x='oplex_score', data=reviews.query('oplex_score < 1 and oplex_score > -1'), ax=ax)
plt.show()

# %%
# Displaying the oplex scores based on the 'reviewText' column
reviews[['reviewText', 'oplex_score']].sample(2)

# %%
# Defining the oplex sentiment from the oplex scores
# And plotting the 'overall' grouping  by oplex sentiment
reviews['oplex_sentiment'] = reviews['oplex_score'].apply(lambda x : "positive"  if x > 0.1
                                                          else ("negative" if x < 0 else "neutral"))
reviews['oplex_sentiment'].value_counts(dropna=False)

sns.countplot(x='overall', hue='oplex_sentiment', data=reviews)

#%%
# Plotting the 'oplex sentiment' against the 'overall' 
sns.boxenplot(x='oplex_sentiment', y='overall', data=reviews)

# %%
# Re-plotting the 'overall' against the 'oplex scores' from the dataset
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
sns.boxenplot(x='overall', y='oplex_score', data=reviews, ax=ax)
plt.show()

# %%
# Defining the true sentiment from the 'overall'
# And defining the y_pred_oplex and y_true_oplex
reviews['oplex_true_sentiment'] = reviews['overall'].apply(lambda x : 
                                                           "positive" if x >= 1 else ("negative"
                                                                                      if x < 0 else "neutral"))
y_true_oplex, y_pred_oplex = reviews['oplex_true_sentiment'].tolist(), reviews['oplex_sentiment'].tolist()

len(y_true_oplex), len(y_pred_oplex)

#%%
# Defining the confusion matrix
from sklearn.metrics import confusion_matrix
oplex_cf_matrix = confusion_matrix(y_true_oplex, y_pred_oplex)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
sns.heatmap(oplex_cf_matrix, fmt='d', square=True, cmap='viridis_r',
            ax=ax, annot=True)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# %%
# Extracting the relevant information from the matrix
# TP, TN, FP, FN
oplex_cf_matrix = list(oplex_cf_matrix.ravel())
oplex_cf_matrix

#%%
## Negative Label Assessment
tp, tn, fp, fn  = 804, 195+199+686+1181, 106+132, 701+495

recall = tp / (tp + fn)
specificity  = tn / (tn + fp)
precision  = tp / (tp + fp)
f1 = (2*tp) / (2*tp + fp + fn)

print("recall : {}\nspecificity : {}\nprecision : {}\nf1 score : {}".format(recall, specificity, precision, f1))

#%%
## Positive Label Assessment
tp, tn, fp, fn  = 1181, 804+701+106+195, 495+199, 132+686

recall = tp / (tp + fn)
specificity  = tn / (tn + fp)
precision  = tp / (tp + fp)
f1 = (2*tp) / (2*tp + fp + fn)

print("recall : {}\nspecificity : {}\nprecision : {}\nf1 score : {}".format(recall, specificity, precision, f1))
# %%
