#%%
# Import the libraries 
import pandas as pd 
import ndjson
import seaborn as sns 

#%%
# Load and read the Amazon reviews dataset
with  open('Automotive_5.json') as f:
    data = ndjson.load(f)

reviews_dataset = pd.DataFrame(data)

reviews_dataset.head()

#%%
# Plot the 'overall' column from the original dataset
sns.countplot(data=reviews_dataset, x='overall', color='green')

# %%
# Count some information values
len(reviews_dataset['asin'].value_counts(dropna=False))

# %%
# Display the shape of the dataset
reviews_dataset.shape

# %%
# Show some information from the dataset
reviews_dataset.info()

# %%
# Undersampling the dataset
one_2000 = reviews_dataset[reviews_dataset['overall'] == 1.0].sample(n=2000)
two_1000 = reviews_dataset[reviews_dataset['overall'] == 2.0].sample(n=1000)
three_1000 = reviews_dataset[reviews_dataset['overall'] == 3.0].sample(n=1000)
four_1000 = reviews_dataset[reviews_dataset['overall'] == 4.0].sample(n=1000)
five_2000 = reviews_dataset[reviews_dataset['overall'] == 5.0].sample(n=2000)

undersampled_reviews = pd.concat([one_2000, two_1000, three_1000, four_1000, five_2000], axis=0)
undersampled_reviews['overall'].value_counts(dropna=False)

# %%
# Plot the 'overall' column from the undersampled_reviews dataset
sns.countplot(data=undersampled_reviews, x='overall')

#%%
# Creating and saving small and big corpus datasets
sample_100k_reviews = reviews_dataset['overall'].sample(n=100000, random_state=42)

undersampled_reviews.to_csv('/Users/jumper/transformers/amazon_reviews/small_corpus.csv')
sample_100k_reviews.to_csv('/Users/jumper/transformers/amazon_reviews/big_corpus.csv')
# %%
