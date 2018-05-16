# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
from keras import Input, Model
from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Permute, Reshape, Lambda, K, RepeatVector, merge, \
    Flatten, BatchNormalization
from keras.preprocessing.sequence import pad_sequences

matplotlib.use('Agg')

MAX_LEN = 80

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


def build_model():
    max_features = 200
    embed_dim = 128
    lstm_dim = 196

    main_input = Input(shape=(MAX_LEN,))

    x = Embedding(max_features, embed_dim, input_length=MAX_LEN)(main_input)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
    attention_mul = attention_3d_block(x)
    x = Flatten()(attention_mul)

    main_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[main_input], outputs=[main_output])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def build_model_no_attention():
    max_features = 200
    embed_dim = 128
    lstm_dim = 196

    main_input = Input(shape=(MAX_LEN,))

    x = Embedding(max_features, embed_dim, input_length=MAX_LEN)(main_input)
    x = Bidirectional(LSTM(lstm_dim, dropout=0.2, recurrent_dropout=0.2))(x)

    main_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[main_input], outputs=[main_output])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def main():
    max_features = 200

    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)

    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)

    epochs = 7

    model = build_model()
    # model = build_model_no_attention()

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    batch_size = 32
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

    print('score: {}'.format(score))
    print('acc: {}'.format(acc))

    attention_vector = np.mean(get_activations(model, X_test, True, 'attention_vec')[0], axis=2).squeeze()
    attention_vector = np.mean(attention_vector, axis=0)

    import matplotlib.pyplot as plt
    import pandas as pd
    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', title='Attention')
    plt.savefig('attention_vec.png')

    attention_vector_indices = np.argsort(attention_vector)[::-1]

    word_index = imdb.get_word_index()
    word_index_inv = {v: k for k, v in word_index.items()}

    with open('attention_word.txt', 'w') as f:
        for i, attention_index in enumerate(attention_vector_indices):
            print('No.{} : {}'.format(i, word_index_inv[attention_index]), file=f)


if __name__ == '__main__':
    main()
