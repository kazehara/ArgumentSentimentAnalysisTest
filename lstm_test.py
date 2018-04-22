# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Dense, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc
from sklearn.model_selection import train_test_split

from aggregate import emotional_rational
from utils import Preprocessor, plot_confusion_matrix, plot_roc_curve


def main():
    emotionals, rationals = emotional_rational()

    preprocessor = Preprocessor()
    emotionals = preprocessor.parse_sentences(emotionals)
    rationals = preprocessor.parse_sentences(rationals)

    emotionals = emotionals[:len(emotionals)]
    rationals = rationals[:len(emotionals)]

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
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
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

    print(classification_report(Y_test[:, 1], np.round(Y_pred[:, 1]), target_names=['rationals', 'emotionals']))

    fpr, tpr, _ = roc_curve(Y_test[:, 1], Y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 'roc.png')

    cnf_matrix = confusion_matrix(Y_test[:, 1], np.round(Y_pred[:, 1]))
    plot_confusion_matrix(cnf_matrix, ['rationals', 'emotionals'], 'cnf.png')


if __name__ == '__main__':
    main()
