# -*- coding: utf-8 -*-
from aggregate import emotional_rational
from utils import Preprocessor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, mean_absolute_error
from keras.utils.np_utils import to_categorical

import numpy as np


def main():
    emotionals, rationals = emotional_rational()

    preprocessor = Preprocessor()
    emotionals = preprocessor.parse_sentences(emotionals)
    rationals = preprocessor.parse_sentences(rationals)

    sentences = emotionals + rationals
    Y = np.array([[0, 1]] * len(emotionals) + [[1, 0]] * len(rationals))

    max_features = 200
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X)

    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    batch_size = 32
    model.fit(X_train, Y_train, epochs=7, batch_size=batch_size, verbose=2)

    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

    print('score: {}'.format(score))
    print('acc: {}'.format(acc))

    Y_pred = model.predict(X_test, batch_size=1, verbose=2)
    print(Y_pred)

    print(mean_absolute_error(Y_test, Y_pred))


if __name__ == '__main__':
    main()
