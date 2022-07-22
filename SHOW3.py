import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

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

filename = 'Trainedmdl.sav'
# load the model from disk
clf = pickle.load(open(filename, 'rb'))

plt.figure(figsize=(10,10))
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test, pred, labels=[0,1])
plot_confusion_matrix(cm, classes=[0,1])
plt.show()

plt.figure(figsize=(10,10))
pred = clf.predict_proba(tfidf_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test.values, pred, pos_label=1)
plt.plot(fpr,tpr,label="{}".format("ACCURACY"))
plt.legend(loc=0)
plt.show()
