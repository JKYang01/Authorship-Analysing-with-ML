
# import the packages 
## encoding and dataframe 
import base64  
import numpy as np 
import pandas as pd 

## use nltk tool to do tokenize 
import nltk
from nltk.corpus import stopwords
import base64
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import string  
import re

## Plotly imports  for visualization
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

##%matplotlib inline is used in ipython to show the visualization plt.show() works too but it has to be added after each figure. 
##%matplotlib inline has the advantage that when it is called once, all figures in the notebook will be inline. 


#import datafile

df = pd.read_csv(r"C:\....whole_data600.csv")  
df.shape


# use the plotly.graph_objs visualize the data distribution of three authors 

z = {'JJ': 'James Joyece', 'RY': 'Richard Yates', 'RC': 'Remond Carvert'}

##input the pramaters

data = [go.Bar(
            x = df.author.map(z).unique(),
            y = df.author.value_counts().values,
            marker= dict(colorscale='Jet',
                         color = df.author.value_counts().values
                        ),
            text='Text entries attributed to Author'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

## make the figture 
fig = go.Figure(data=data, layout=layout)  

py.iplot(fig, filename='basic-bar')

# use regular expression and funtion to do simple feature engineering to get some feature directly #

## get stopword list from nltk corpus
eng_stopwords = set(stopwords.words("english")) 
stemmer = EnglishStemmer()
pd.options.mode.chained_assignment = None

###Using NLTK to extract unique words unisng the feauture engeering re.sub 
### replace anything that not a word and stopwords are not included 
###the punctuation is not removed just using NLTK tokenizer 
###using word.isalpha() in if to make sure it is a word

df['split'] = df['text'].apply(nltk.word_tokenize)
df['words'] = df['split'].apply(lambda x : [word.lower() for word in x if word.isalpha()])
df['stemmed'] = df['words'].apply(lambda x: [stemmer.stem(y) for y in x 
                                                         if re.sub('[^a-z]+','',y.lower()) not in eng_stopwords])
# count the number of unique words and stop words in the text #

df["num_unique_words"] = df['stemmed'].apply(lambda x: len(x))
df["num_stopwords"] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

# visualize results#
plt.figure(figsize=(12,8))

## for single figure  
df['num_unique_words'].loc[df['num_unique_words']>600] = 600 
###  use seaborn .violinplot() fuction to visualize the distribution of unique words number in each document
sns.violinplot(x='author', y='num_words', data=train_df)  
###  set the pramater in plotly
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15)
plt.show()
###truncation for better visuals, set the range if the number of words is more than 600 it will be regard as 600
### .loc[]Access a group of rows and columns by label(s) or a boolean array.
###.loc[] is primarily label based, but may also be used with a boolean array. 
###https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.loc.html


## for multipule fiugres use f, axes  
f, axes = plt.subplots(2,1,figsize=(21, 18), sharex=True, sharey = False)  
###matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, 
###                           subplot_kw=None, gridspec_kw=None, **fig_kw)[source]
###sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
###Controls sharing of properties among x (sharex) or y (sharey) 
###axes: True or 'all': x- or y-axis will be shared among all subplots.
###https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html


## figure of unique words
###truncation for better visuals set if x>600 regard as = 600
df['num_unique_words'].loc[df['num_unique_words'] >600] = 600 
sns.violinplot(x='author', y='num_unique_words', data=df, ax = axes[0])
## figure of stop words
df['num_stopwords'].loc[df['num_stopwords'] >600] = 600                    
sns.violinplot(x='author', y='num_stopwords', data=df, ax  = axes[1])
sns.despine(left=True)

