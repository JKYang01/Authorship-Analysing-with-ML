# IST-Text-Mining-Project
# GLOOMY AUTHOR IDENTIFICATION
Authorship identification with simple feature engineering and LinearSVC to find writing style on word-level 
and extract the feature words by employing the algorithm
Using the package of Matplotlib --plotly to visualize the results and provide more insights

## Plotly:
Plotly is an online collaborative data analysis and graphing tool. The Python API allows you to access all of Plotly's 
functionality from Python. Plotly figures are shared, tracked, and edited all online and the data is always accessible from the graph. 
plotly · PyPI : https://pypi.org/project/plotly/

## The Dataset 
Three books <Dubliners>(by James.Joyce) <Eleven Kinds of loneliness>(by Richard Yates) <What we talk about when we talk about love> (by Raymond Carver)
Using regular expression to clean the files in notpad++ 
Then mixed them and seperated into two types of .csv files：
word_data600.csv
Sentence_data.csv
each one contains two columns ‘author’ and  ‘text’
The differences between the two files are the unit of words:
word_data600.csv, each row contains 600 words. 
Sentence_data.csv, each document has only one sentence. 
The distribution of author in the data is:
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/%E5%9B%BE%E7%89%871.png)

## The code of writing files 
firtly split the whole text into words with .split() 
then use .iter() and zip() to sperate the long list of words into sublists and each  contains 600 words  
```ruby
def list_of_words(init_list,sub_list_len):
    list_of_word = list(zip(*(iter(init_list),)*sub_list_len)) 
    end_list = [list(i) for i in list_of_word]
    ....
    return end_list
new_word_list = list_of_words(unit_JJ,600)
```

## Simple feature engeneering 
quite a few special characters which might be good features.  Get directly in the text data without complex calculation. 
### Text-based features:  frequency of specific words
By employing NLTK tags to catagrize the words and count them in each group
```ruby
def fraction_noun(x):
    """function to give us fraction of noun over total words """
    .....
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)
```
Average fraction of noun of JJ:  0.25955366897627385 
Average fraction of noun of RY:  0.2410677319129729 
Average fraction of noun of RC:  0.23489160784172405 

Average fraction of adjective of JJ:  0.06259012981023131
Average fraction of adjective of RY:  0.06450037298938273
Average fraction of adjective of RC:  0.04310757722957404 

Average fraction of verb of JJ:  0.19481202119363242
Average fraction of verb of RY:  0.19127114870798473
Average fraction of verb of RC:  0.22169705750928081 

Average fraction of adverb of JJ:  0.052373569165498554
Average fraction of adverb of RY:  0.060251000997142104
Average fraction of adverb of RC:  0.05071981984771416

Distribution of the Number of unique words and stopwords in each 600 words group:
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/download.png)

### Meta features:  number of words of each sentence:
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/%E5%9B%BE%E7%89%873.png)
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/%E5%9B%BE%E7%89%872.png)

## Get feature words with LinerSVC
Feature words for Prediction authorship

Term Frequency vectorizer to vectorize the train_data separated from the word_data600.csv with the following parameter.
min_df = 5 stopwords=list（stopwords） enconding = 'latin1'  The resutlt:
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/%E5%9B%BE%E7%89%876.png)

### Feature words:
James Joyce “Dubliner”:
'confused' 'head' 'body' 'people' 'street' 'round' 'air' 'slowly' 'began' yes' 'gone' 'money' 'tried' 'frank' 'women' 'priest' 'dead' 'saying' 'grey' 'heart' 'shop' 'felt' 'tea' 'used' 'death' 'eyes' 'father' 'falling' 'spoke' 'life' 'live' 'great' 'dark' 'evening' 'end' 'young' 'poor' ‘said aunt' 

Richard Yates “Eleven Kinds of Loneliness”:
'living room' 'rock' 'started' 'bathroom' 'moved' 'sit' 'morning' 'looked' 'comes' 'bed' 'fish' 'coffee' 'chair' 'son' 'ashtray' 'shot' 'saw' 'took' 'clean' 'hear' 'picked' 'place' 'days' 'thinking' 'green' 'sofa' anymore' 'say' 'kept' 'guard' 'hooks' 'newspaper' 'watched' 'dad' 'things' 'terri' 

Raymond Carver “What We Talk about When Talk about Love”:
'everybody' 'want' 'bag' 'cigarette' 'won' 'platoon' 'army' 'best' 'writer' 'easy' 'week' 'trouble' 'got' 'deal' 'real' 'later' 'book' 'day' 'smile' 'jean' 'job' 'miss price' 'price’ 'lips' 'mcintyre' 'time' steps' 'ralph' 'new' 'tiny' 'guess' 'sure' 'building' 'kind' 'oh' 'thing' 'way' ‘reer’

## Other features –The author distribution in top 50 popular words
the progressing:
using NLTK tokenize (stemming, remove stopwords …) → Built vocabulary
Count → count word overall  and  by author  → 50 highest frequency
### Using Plotly  to visualize
Fuction of dash lines:
df['ALL'] * word_count[authors[0]].sum() / word_count['ALL'].sum()
df['ALL'] * word_count[authors[:2]].values.sum() / word_count['ALL'].sum() 
![alt text](https://github.com/JKYang01/IST-Text-Mining-Porject/blob/master/IST736%20project/%E5%9B%BE%E7%89%875.png)



