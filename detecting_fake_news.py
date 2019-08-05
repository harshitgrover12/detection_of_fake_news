# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 23:09:27 2019

@author: ravik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import VotingClassifier

#importing dataset
dataset=pd.read_csv('fake_or_real_news.csv')




#for encoding label to 0 and 1
label=LabelEncoder()
dataset['label']=label.fit_transform(dataset['label'])
y=dataset['label']



#cleaning the text 
corpus=[]
for i in range(0,6335):
    statement=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    statement=statement.lower()
    statement=statement.split()
    ps=PorterStemmer()
    statement=[ps.stem(word) for word in statement if not word in stopwords.words('english')]
    statement=' '.join(statement)
    corpus.append(statement)








#feature engineering and using bag of words model
# create bag-of-words
all_words = []

for message in corpus:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:1500]


#function to find features in a message
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(corpus[0])
for key, value in features.items():
    if value == True:
        print(key)
# Now lets do it for all the messages
messages = zip(corpus, y)
messages=list(messages)
# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages] 
#test set of 33% and training set of 67%
training, testing = train_test_split(featuresets, test_size = 0.33, random_state=seed)



# Using sklearn algorithms in NLTK


model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))   
#    





# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))
    
# Ensemble methods - Voting classifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]


models = zip(names, classifiers)
models=list(models)
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))  

#out of all these classifiers max accuracy achieved using logistic regession  of 86.22