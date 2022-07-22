import tkinter
import PyQt5.QtWidgets as QtWidgets
from tkinter import filedialog
from tkinter import messagebox 
from PyQt5 import QtCore, QtGui , QtWidgets


import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
import pickle

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
#---------------------
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


def false(key):
    
    #input=input('Enter Review: ')
    input = [key]

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
        print('Review is Normal \n')
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Review is TRUE")
        msg.setWindowTitle("Review Status")
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        #self.close()


#---------------------

        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(main.jpg);\n"""))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(780, 180, 151, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(255, 255, 255);"))
       
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################

        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(100, 100, 370, 21))
        self.text.setText("Review Test")
        self.text.setStyleSheet(_fromUtf8("color: rgb(255, 255, 255);"))
        
        self.uname_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.uname_lineEdit.setGeometry(QtCore.QRect(100, 130, 370, 30))
        self.uname_lineEdit.setText(_fromUtf8(""))
        self.uname_lineEdit.setObjectName(_fromUtf8("uname_lineEdit"))
        self.uname_lineEdit.setStyleSheet(_fromUtf8("background-color: rgb(155, 128, 0);\n"
"color: rgb(255, 255, 255);"))

        #------------------------------------------------------------------------------
        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(100, 170, 370, 20))
        self.text.setText("Review URL")
        self.text.setStyleSheet(_fromUtf8("color: rgb(255, 255, 255);"))
        
        self.l1 = QtWidgets.QLineEdit(self.centralwidget)
        self.l1.setGeometry(QtCore.QRect(100, 200, 370, 30))
        self.l1.setText(_fromUtf8("https://www."))
        self.l1.setObjectName(_fromUtf8("uname_lineEdit"))
        self.l1.setStyleSheet(_fromUtf8("background-color: rgb(155, 128, 0);\n"
"color: rgb(255, 255, 255);"))


        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(100, 240, 370, 20))
        self.text.setText("Review Source")
        self.text.setStyleSheet(_fromUtf8("color: rgb(255, 255, 255);"))
        
        self.l2 = QtWidgets.QLineEdit(self.centralwidget)
        self.l2.setGeometry(QtCore.QRect(100, 270, 370, 30))
        self.l2.setText(_fromUtf8(""))
        self.l2.setObjectName(_fromUtf8("uname_lineEdit"))
        self.l2.setStyleSheet(_fromUtf8("background-color: rgb(155, 128, 0);\n"
"color: rgb(255, 255, 255);"))
        #------------------------------------------------------------------------------

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 180, 211, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(255, 255, 255);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 220, 211, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(255, 255, 255);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(550, 260, 211, 27))
        self.pushButton_5.clicked.connect(self.show3)
        self.pushButton_5.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(255, 255, 255);"))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "FALSE REVIEW PREDICTION", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        self.pushButton_5.setText(_translate("MainWindow", "PERFORMANCE ANALYSIS", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        self.close()
         
    def show1(self):
        username1=self.uname_lineEdit.text()
        print("TEST = "+username1)
        false(username1)
        #import SHOW1               

    def show2(self):
        import SHOW2
        
    def show3(self):
        import SHOW3                    
                


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())


