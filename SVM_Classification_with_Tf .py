# import data

import pandas as pd
train=pd.read_csv(r"C:\Users\wangtao\Desktop\train_data600.csv",encoding='latin1')
y_train=train['author'].values
X_train=train['text'].values
test=pd.read_csv(r"C:\Users\wangtao\Desktop\test_data600.csv",encoding='latin1')
y_test=test['author'].values
X_test=test['text'].values

# test the imported data
y_train.shape

# get lables
import numpy as np
training_labels = set(y_train)
print(training_labels)
#from scipy.stats import itemfreq
#test_category_dist = itemfreq(y_test)
#print(test_category_dist)
training_category_dist = np.unique(y_train)
print(training_category_dist)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
## how to add in my own stopword list https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/37679
from sklearn.feature_extraction import text 

## Using my own stop words list
stop_words_list = [x.strip() for x in open(r'C:\Users\wangtao\Desktop\stopwords.txt','r').read().split(',')]

##another way to add in stop word list using  .union()
##stop_words_add = text.ENGLISH_STOP_WORDS.union(['mr','dr','martin','mary','jane','julia','lily',
    ##'maria','jimmy','keran','dillion','mr fixit',
    ##'mrs','mr browne','ll'])
    
bigram_tf = CountVectorizer(min_df=5, stop_words=list(stop_words_list),ngram_range=(1,2),encoding='latin1')

# The vectorizer can do "fit" and "transform"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()
X_train_vec = bigram_tf.fit_transform(X_train)
#X_train_tfidfvec = bigram_tfidf.fit_transform(X_train)

# import the algorithm
from sklearn.svm import LinearSVC
svm_clf = LinearSVC(C=1)
svm_clf.fit(X_train_vec,y_train)

# rank the feature words
feature_rank1= sorted(zip(svm_clf.coef_[1],bigram_tf.get_feature_names()))
a1=feature_rank1[-50:]
for i in range(0,50):
    print(a1[i])

# get 50 feature words of each author
feature_rank = sorted(zip(svm_clf.coef_[0],bigram_tf.get_feature_names()))
feature_rank1= sorted(zip(svm_clf.coef_[1],bigram_tf.get_feature_names()))
feature_rank2 = sorted(zip(svm_clf.coef_[2],bigram_tf.get_feature_names()))

a=feature_rank[-50:]
a1=feature_rank1[-50:]
a2=feature_rank2[-50:]
l=[]
l1=[]
l2=[]
for i in range(0,50):
    print(a2[i])
    l.append(a[i])
    l1.append(a1[i])
    l2.append(a2[i])
f_data = {'James Joyce':l,'Raymond Carver':l1,'Richard Yates':l2}

# build the dataframe of the results and write into csv file
df_f_data=pd.DataFrame(f_data)
with open (r'C:\Users\wangtao\Desktop\featurewords.csv','w')as f:
    df_f_data.to_csv(f)


# the scores that indicate the performance of the model
X_test_vec = bigram_tf.transform(X_test)
svm_clf.score(X_test_vec,y_test)
from sklearn.metrics import confusion_matrix
y_pred = svm_clf.predict(X_test_vec)
cm = confusion_matrix(y_test,y_pred,labels=['JJ','RY','RC'])
print(cm)

# The scores in confusion matrix
from sklearn.metrics import classification_report
target_names = ['JJ','RY','RC']
print(classification_report(y_test,y_pred,target_names=target_names))


# print out specific type of error for further analysis
# print out the very positive examples that are mistakenly predicted as negative
# according to the confusion matrix, there should be 53 such examples
# note if you use a different vectorizer option, your result might be different
err_cnt = 0
for i in range(0, len(y_test)):
    if(y_test[i]=='RY' and y_pred[i]=='RC'):
        print(X_test[i])
        print(i)
        err_cnt = err_cnt+1
print("errors:", err_cnt)
