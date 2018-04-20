# -*- coding: utf-8 -*-
from aggregate import emotional_rational
from utils import Preprocessor
from gensim.models import word2vec


def main():
    emotionals, rationals = emotional_rational()

    preprocessor = Preprocessor(ignore_parts=['助詞', '助動詞'])
    emotionals = preprocessor.parse_sentences(emotionals)
    rationals = preprocessor.parse_sentences(rationals)

    sentences = emotionals + rationals

    sentences = list(map(lambda x: x.split(' '), sentences))

    model = word2vec.Word2Vec(sentences, size=100, min_count=5, window=5, iter=3)
    model.save('dataset/collagree.word2vec.gensim.model')

    print(model.most_similar('名古屋'))


if __name__ == '__main__':
    main()