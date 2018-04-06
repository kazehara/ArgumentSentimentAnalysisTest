# -*- coding: utf-8 -*-
from typing import List

from aggregate import emotional_rational
from sklearn.feature_extraction.text import CountVectorizer
import MeCab


class NBSVMClassifier:

    def __init__(self):
        self.mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    def _parse(self, text: str) -> List[str]:
        return self.mecab.parse(text)

    def _analyzer(self, text: str) -> List[str]:
        return self._parse(text)

    def vectorize(self, sentences: List[str]):
        pass


def main():
    emotionals, rationals = emotional_rational()

    # vectorizer = CountVectorizer(ngram_range=(1, 3), binary=True)


if __name__ == '__main__':
    main()
