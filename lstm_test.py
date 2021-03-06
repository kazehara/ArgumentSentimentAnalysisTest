# -*- coding: utf-8 -*-
import matplotlib
from keras.engine import Layer

matplotlib.use('Agg')

from emotional_dict import EmotionalDict
from features import AdditionalFeatures

import numpy as np
from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Permute, Reshape, Lambda, K, RepeatVector, merge, \
    Flatten, BatchNormalization, Dropout, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc
from sklearn.model_selection import train_test_split

from aggregate import emotional_rational
from utils import Preprocessor, plot_confusion_matrix, plot_roc_curve

MAX_LEN = 150

SINGLE_ATTENTION_VECTOR = False


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    input = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([input] + [K.learning_phase()], [output]) for output in outputs]
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # (batch_size, max_len, embed_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, MAX_LEN))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(MAX_LEN, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def build_model(add_features_shape):
    max_features = 200
    embed_dim = 128
    lstm_dim = 196

    main_input = Input(shape=(MAX_LEN,), name='main_input')

    x = Embedding(max_features, embed_dim, input_length=MAX_LEN, trainable=True)(main_input)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)

    # Additional Features
    additional_input = Input(shape=(MAX_LEN,), name='add_input')
    t_additional_input = RepeatVector(MAX_LEN)(additional_input)
    t_additional_input = Dense(256, activation='tanh')(t_additional_input)
    t_additional_input = Dropout(0.5)(t_additional_input)

    x = concatenate([x, t_additional_input])

    # Attention
    attention_mul = attention_3d_block(x)
    x = Flatten()(attention_mul)

    main_output = Dense(2, activation='softmax')(x)

    model = Model(inputs=[main_input, additional_input], outputs=[main_output])
    # model = Model(inputs=[main_input], outputs=[main_output])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


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
    X = pad_sequences(X, maxlen=MAX_LEN)

    epochs = 15

    # --- Add Features ---
    dict_loader = EmotionalDict('dataset/nouns', 'dataset/verbs')
    emotional_dict = dict_loader.load()

    features_loader = AdditionalFeatures(emotionals+rationals, emotional_dict)
    add_features = features_loader.emotional_features()
    ######################

    x_aux_train = add_features[:848]
    x_aux_test = add_features[848:]

    model = build_model(x_aux_train.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    batch_size = 32
    model.fit({'main_input': X_train, 'add_input': x_aux_train}, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    score, acc = model.evaluate({'main_input': X_test, 'add_input': x_aux_test}, Y_test, verbose=2, batch_size=batch_size)

    print('score: {}'.format(score))
    print('acc: {}'.format(acc))

    Y_pred = model.predict({'main_input': X_test, 'add_input': x_aux_test}, batch_size=1, verbose=2)

    print(classification_report(Y_test[:, 1], np.round(Y_pred[:, 1]), target_names=['rationals', 'emotionals']))

    fpr, tpr, _ = roc_curve(Y_test[:, 1], Y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 'roc.png')

    cnf_matrix = confusion_matrix(Y_test[:, 1], np.round(Y_pred[:, 1]))
    plot_confusion_matrix(cnf_matrix, ['rationals', 'emotionals'], 'cnf.png')

    attention_vector = np.mean(get_activations(model, X_test, True, 'attention_vec')[0], axis=2).squeeze()
    attention_vector = np.mean(attention_vector, axis=0)

    import matplotlib.pyplot as plt
    import pandas as pd
    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', title='Attention')
    plt.savefig('attention_vec.png')

    attention_vector_indices = np.argsort(attention_vector)[::-1]

    word_index = tokenizer.word_index
    word_index_inv = {v: k for k, v in word_index.items()}

    with open('attention_word.txt', 'w') as f:
        for i, attention_index in enumerate(attention_vector_indices, start=1):
            try:
                print('No.{} : {}'.format(i, word_index_inv[attention_index]), file=f)
            except KeyError:
                continue


if __name__ == '__main__':
    main()
