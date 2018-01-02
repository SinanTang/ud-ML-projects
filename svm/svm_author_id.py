#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
# sys.path.append("../tools/")
from ..tools.email_preprocess import preprocess
# $ python3 -m ud-ML-projects.svm.svm_author_id


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
def SVMaccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.svm import SVC
    # clf = SVC(kernel="linear")
    clf = SVC(C=10000., kernel="rbf") # 1k+ s when trained on full dataset..
    # clf = SVC()

    # # cutting down on training dateset
    # features_train = features_train[:len(features_train)//100]
    # labels_train = labels_train[:len(labels_train)//100]

    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time() - t0), "s")
    # linear: 161s / 0s

    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time() - t1), "s")
    # linear: 16s / 1s

    # print(pred[10], pred[26], pred[50])

    # how many Chris emails (lable 1) detected?
    print("Sum:", sum(pred))

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc
    # linear: 0.98 / 0.88
    # rbf + full data: 0.99

print("SVM accuracy score:", SVMaccuracy(features_train, labels_train, features_test, labels_test))

#########################################################
