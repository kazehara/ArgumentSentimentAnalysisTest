# -*- coding: utf-8 -*-
from keras.datasets import imdb


def main():
    word_index = imdb.get_word_index()

    word_index_inv = {v: k for k, v in word_index.items()}

    print(word_index_inv)


if __name__ == '__main__':
    main()
