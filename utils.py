# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import MeCab
from typing import List


class Preprocessor:

    def __init__(self):
        self.mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        self.X_train = None
        self.y_train = None

    def _parse(self, text: str) -> str:
        text = self.mecab.parse(text)
        text = text.strip()
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
