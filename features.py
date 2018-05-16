# -*- coding: utf-8 -*-
from typing import List, Set, Dict

from aggregate import emotional_rational
from emotional_dict import EmotionalDict

import numpy as np

from utils import Preprocessor


MAX_LEN = 150


class AdditionalFeatures:

    def __init__(self, sentences: List[str], emotional_dict: Dict[str, Set[str]]) -> None:
        # sentences like : [['今日 は 楽しい'], ['明日 は 雨']]
        self.sentences = []
        for sentence in sentences:
            self.sentences.append(sentence.split(' '))
        self.emotional_dict = emotional_dict

    def emotional_features(self) -> np.array:
        features = []

        for sentence in self.sentences:
            feature = []
            for word in sentence[:MAX_LEN]:
                if word in self.emotional_dict['positive']:
                    feature.append(1)
                elif word in self.emotional_dict['negative']:
                    feature.append(-1)
                else:
                    feature.append(0)
            if len(sentence) < MAX_LEN:
                for _ in range(MAX_LEN - len(sentence)):
                    feature.append(0)

            if len(feature) != MAX_LEN:
                print(len(feature))
                raise Exception

            features.append(feature)

        features = np.asarray(features).reshape((len(self.sentences), MAX_LEN))

        print(features.shape)

        return features


def test():
    dict_loader = EmotionalDict('dataset/nouns', 'dataset/verbs')
    emotional_dict = dict_loader.load()

    emotionals, rationals = emotional_rational()

    preprocessor = Preprocessor()
    emotionals = preprocessor.parse_sentences(emotionals)
    rationals = preprocessor.parse_sentences(rationals)

    corpus = emotionals + rationals

    features_loader = AdditionalFeatures(corpus, emotional_dict)
    features = features_loader.emotional_features()

    print(features)


if __name__ == '__main__':
    test()
