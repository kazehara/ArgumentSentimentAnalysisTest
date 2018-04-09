# -*- coding: utf-8 -*-
from typing import List, Any

from aggregate import emotional_rational
from sklearn.feature_extraction.text import CountVectorizer
import MeCab
from nbsvm.nbsvm import NBSVM
from sklearn.metrics import f1_score, roc_curve, auc

import numpy as np


class NBSVMClassifier:

    def __init__(self):
        self.mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        self.X_train = None
        self.y_train = None

    def _preprocess(self, text: str) -> str:
        text = text.strip()
        return text

    def _parse(self, text: str) -> str:
        text = self.mecab.parse(text)
        text = text.strip()
        return text

    def parse_sentences(self, sentences: List[str]) -> List[str]:
        return list(map(self._parse, sentences))

    def vectorize(self, corpus: List[str]):
        vectorizer = CountVectorizer()
        self.X_train = vectorizer.fit_transform(corpus)


def main():
    emotionals, rationals = emotional_rational()

    classifier = NBSVMClassifier()
    emotionals = classifier.parse_sentences(emotionals)
    rationals = classifier.parse_sentences(rationals)

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
    print('AUC of emotionals : {}'.format(auc(fpr, tpr)))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=0)
    print('AUC of rationals : {}'.format(auc(fpr, tpr)))


if __name__ == '__main__':
    main()
