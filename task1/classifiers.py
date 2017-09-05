# -*- coding: utf-8 -*-

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

def get_classifier_by_name(name, n_jobs):
    if name.lower() == 'et':
        return ExtraTreesClassifier(n_estimators=50, max_features='auto', class_weight='balanced', n_jobs=n_jobs)
    elif name.lower() == 'linearsvc':
        return MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs=n_jobs)
    elif name.lower() == 'mlp':
        return MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu'), n_jobs=n_jobs)
    elif name.lower() == 'bayes':
        return MultiOutputClassifier(MultinomialNB())
    else:
        print("Unkown classifier: %s" % name)
        exit()
 
