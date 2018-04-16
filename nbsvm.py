# -*- coding: utf-8 -*-
from typing import List, Any

from aggregate import emotional_rational
from sklearn.feature_extraction.text import CountVectorizer
import MeCab
from nbsvm.nbsvm import NBSVM
from sklearn.metrics import f1_score, roc_curve, auc
from utils import plot_roc_curve, Preprocessor

import numpy as np


def main():
    emotionals, rationals = emotional_rational()

    preprocessor = Preprocessor()
    emotionals = preprocessor.parse_sentences(emotionals)
    rationals = preprocessor.parse_sentences(rationals)

    train_pos = emotionals[:len(emotionals)//2]
    train_neg = rationals[:len(rationals)//2]

    test_pos = emotionals[len(emotionals)//2:]
    test_neg = rationals[len(rationals)//2:]

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(train_pos + train_neg)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    X_test = vectorizer.transform(test_pos + test_neg)
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    print('Vocabulary size : {}'.format(len(vectorizer.vocabulary_)))

    nbsvm = NBSVM()
    nbsvm.fit(X_train, y_train)

    print('Test accuracy : {}'.format(nbsvm.score(X_test, y_test)))

    y_pred = nbsvm.predict(X_test)
    print('F1 score : {}'.format(f1_score(y_test, y_pred, average='macro')))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('AUC of emotionals : {}'.format(roc_auc))
    plot_roc_curve(fpr, tpr, roc_auc, 'nbsvm_emotional_roc.png')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print('AUC of rationals : {}'.format(roc_auc))
    plot_roc_curve(fpr, tpr, roc_auc, 'nbsvm_rational_roc.png')


if __name__ == '__main__':
    main()
