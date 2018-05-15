# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import MeCab
from typing import List, Optional
import numpy as np
import itertools


class Preprocessor:

    def __init__(self, ignore_parts: List[str] = []):
        self.mecab = MeCab.Tagger('-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd')
        self.mecab.parse('')
        self.X_train = None
        self.y_train = None
        self.ignore_parts = ignore_parts

    def _parse(self, text: str) -> str:
        # text = self.mecab.parse(text)
        # text = text.strip()

        origin = []

        node = self.mecab.parseToNode(text)
        while node:
            feature = node.feature.split(',')[0]
            if feature not in self.ignore_parts:
                origin.append(node.surface)
            node = node.next

        text = ' '.join(origin)

        return text

    def parse_sentences(self, sentences: List[str]) -> List[str]:
        return list(map(self._parse, sentences))


def plot_roc_curve(fpr, tpr, roc_auc, output_filename):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(output_filename)


def plot_confusion_matrix(cm, classes, output_filename, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(output_filename)
