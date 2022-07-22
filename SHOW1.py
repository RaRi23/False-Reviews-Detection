import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import ctypes
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *


df = pd.read_csv('Data.csv')
# use a Pandas DataFrame and check the shape, head and apply any necessary transformations.
print(df.shape)
print('PRINT HEAD',df.head())

# TARGET:Label Coloumn
# separate the labels
y = df['Rev_Type']
X=df['Review_Title']
# Extracting the training data
# Remove Target Coloumn
# set up training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)


print('------------------TF IDF -----------------')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
tfidf_test = tfidf_vectorizer.transform(X_train)
y_test=y_train

import pickle

# save the model to disk
filename = 'Trainedmdl.sav'
# load the model from disk
clf = pickle.load(open(filename, 'rb'))


input=input('Enter Review: ')
input = [input]

TEST = tfidf_vectorizer.transform(input)
pred = clf.predict(TEST)
if pred ==1:
    print('Review is False \n')
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("Review is FALSE")
    msg.setWindowTitle("Review Status")
    msg.setStandardButtons(QMessageBox.Ok)
    retval = msg.exec_()
    #self.close()
    
else:
    print('Review is True \n')
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("Review is NORMAL")
    msg.setWindowTitle("Review Status")
    msg.setStandardButtons(QMessageBox.Ok)
    retval = msg.exec_()
    #self.close()
        
