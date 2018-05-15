# -*- coding: utf-8 -*-
from keras.datasets import imdb


def main():
    word_index = imdb.get_word_index()

    print(word_index)


if __name__ == '__main__':
    main()
