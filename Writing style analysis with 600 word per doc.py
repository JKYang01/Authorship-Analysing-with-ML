# IST-Text-Mining-Porject
Authorship identification with simple feature engineering and LinearSVC to find writing style on word-level and extract the feature words by employing the algorithm

## import the packages 

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

#%matplotlib inline is used in ipython to show the visualization plt.show() works too but it has to be added after each figure. 
#%matplotlib inline has the advantage that when it is called once, all figures in the notebook will be inline. 


## import datafile

df = pd.read_csv(r"C:\....whole_data600.csv")  
df.shape


## use the plotly.graph_objs visualize the data distribution of three authors 

z = {'JJ': 'James Joyece', 'RY': 'Richard Yates', 'RC': 'Remond Carvert'}

###input the pramaters

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

### make the figture 
fig = go.Figure(data=data, layout=layout)  

py.iplot(fig, filename='basic-bar')

## use regular expression and funtion to do simple feature engineering to get some feature directly
