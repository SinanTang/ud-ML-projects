#!/usr/bin/python
"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
# from os import path
from time import time
# sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from ..tools.email_preprocess import preprocess
## to deal with (deprecated) relative imports in Python 3.x,
## run this shell line outside the ud120-projects dir:
# $ python3 -m ud-ML-projects.naive_bayes.nb_author_id

### features_train and features_test are the features for
### the training and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time() - t0), "s")
    # 2s

    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time() - t1), "s")
    # 0s

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    return accuracy
    # 0.97

print("Accuracy Score:", NBAccuracy(features_train, labels_train, features_test, labels_test))

#########################################################
