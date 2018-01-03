#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
# sys.path.append("../tools/")
from ..tools.email_preprocess import preprocess
# $ python3 -m ud-ML-projects.decision_tree.dt_author_id

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# number of features selected
# print(len(features_train[0]))

#########################################################
### your code goes here ###
def DTAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=40)

    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0), "s")
    # 40: 71s

    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1), "s")
    # 40: 0s

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels_test, pred)
    return acc
    # 40: 0.978

print("DT accuracy score:", DTAccuracy(features_train, labels_train, features_test, labels_test))

#########################################################
