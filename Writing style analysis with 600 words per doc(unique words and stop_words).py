
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

# Counting and caculate the fraction of specific words in each group
# the fraction of noun
def fraction_noun(x):
    """function to give us fraction of noun over total words """
    text_splited = str(x).split()
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = len(text_splited)
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)
# the fraction of adjective
def fraction_adj(x):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = str(x).split()
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = len(text_splited)
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

# the fraction of verb
def fraction_verb(x):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = str(x).split()
    text_splited = [''.join(c for c in s if c not in string.punctuation)for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = len(text_splited)
    pos_list = nltk.pos_tag(text_splited)
    verb_count = len( [w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ') ])
    return(verb_count/word_count)

# the fraction of adverb
def fraction_adv(x):
    text_splited = str(x).split()
    text_splited = [''.join(c for c in s if c not in string.punctuation)for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = len(text_splited)
    pos_list  = nltk.pos_tag(text_splited)
    adv_count = len([w for w in pos_list if w[1] in ('RB','RBR','RBS')])
    return(adv_count/word_count)

# Using lambda to apply the fuctions to each document 并新建一个 column 去存储结果
df['fraction_noun'] = df['text'].apply(lambda x:fraction_noun(x))
df['fraction_adj'] = df['text'].apply(lambda x:fraction_adj(x))
df['fraction_verb'] = df['text'].apply(lambda x:fraction_verb(x))
df['fraction_adv'] = df['text'].apply(lambda x: fraction_adv(x))

#Print the results  using .loc[] to 选定需要操作的区域 and use .mean()计算平均数
## the average fration of noun
print('Average fraction of noun of JJ: ',df['fraction_noun'].loc[df['author']=='JJ'].mean())
print('Average fraction of noun of RY: ',df['fraction_noun'].loc[df['author']=='RY'].mean())
print('Average fraction of noun of RC: ',df['fraction_noun'].loc[df['author']=='RC'].mean(),'\n')
## the average fration of adjective
print('Average fraction of adjective of JJ: ',df['fraction_adj'].loc[df['author']=='JJ'].mean())
print('Average fraction of adjective of RY: ',df['fraction_adj'].loc[df['author']=='RY'].mean())
print('Average fraction of adjective of RC: ',df['fraction_adj'].loc[df['author']=='RC'].mean(),'\n')
##the average fraction of verb
print('Average fraction of verb of JJ: ',df['fraction_verb'].loc[df['author']=='JJ'].mean())
print('Average fraction of verb of RY: ',df['fraction_verb'].loc[df['author']=='RY'].mean())
print('Average fraction of verb of RC: ',df['fraction_verb'].loc[df['author']=='RC'].mean(),'\n')
##the average fraction of adverb
print('Average fraction of adverb of JJ: ',df['fraction_adv'].loc[df['author']=='JJ'].mean())
print('Average fraction of adverb of RY: ',df['fraction_adv'].loc[df['author']=='RY'].mean())
print('Average fraction of adverb of RC: ',df['fraction_adv'].loc[df['author']=='RC'].mean())


# Count word again and see the distribution of each author in the top 50 popular words
# Count words overall and by author

#built dictionaries (for counting by author)
word_count = {'ALL': pd.Series([y for x in df['stemmed'] for y in x]).value_counts()} #make a dictionary of all data
authors = ['JJ','RY','RC']

# enumerate() function  count the number of authors in each word by using auth:i dictionary
authors_dict = {auth: i for i, auth in enumerate(authors)} 
for auth in authors:
    word_count[auth] = pd.Series([y for x in df.loc[df['author']==auth, 'stemmed']
                                  for y in x]).value_counts()
word_count = pd.DataFrame(word_count).fillna(0).astype(int).sort_values('ALL', ascending=False)[['ALL']+authors]

print(word_count[authors[:3]])
print('Count for the most common words (excl. stopwords)')

# Visialize the distribution 
plt.style.use('ggplot')
plt.rcParams['font.size'] = 16
plt.figure(figsize=(20,10))
bottom = np.zeros((50))
ind = np.arange(50)
df = word_count.head(50)
for auth in authors:
    # Stacked bar with actual numbers.
    # Uncomment the below for percentages instead.
    vals = df[auth]# / df['ALL']
    plt.bar(ind, vals, bottom=bottom, label = auth)
    bottom += vals

# If using percentages, replace the two "df['ALL']" by "np.ones(df['ALL'].shape)"
plt.plot(ind, df['ALL'] * word_count[authors[0]].values.sum() / word_count['ALL'].sum(), 'k--',
         label='Expected cutoffs for\n uninformative words')
plt.plot(ind, df['ALL'] * word_count[authors[:2]].values.sum() / word_count['ALL'].sum(), 'k--', label='')
plt.xticks(ind, df.index, rotation='vertical')
#plt.yticks(np.arange(0,1.1,0.2), ['{:.0%}'.format(x) for x in np.arange(0,1.1,0.2)])
plt.legend(fontsize=24)
plt.title('Top 50 word count split by author (dotted lines is the global average)', fontsize=24)
plt.xlim([-0.7,49.7])## 设置坐标轴取值范围

