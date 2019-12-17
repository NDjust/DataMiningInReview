# import Tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

# import keras
from keras.models import Sequential
# CNN
from keras.layers import Dense, Conv1D, Dropout, GlobalMaxPool1D
# RNN
from keras.layers import Embedding, LSTM, Bidirectional
# Radam optimization.
from keras_radam.training import RAdamOptimizer


def select_optimizer(optimizer):
    """ Select Optimizer.

    Select optimizer in rmsprop, adam, Radm.

    :param optimizer: input select optimizer
    :return: selected optimizer
    """
    if optimizer == 'RMS':
        optimizer = optimizers.RMSprop(lr=0.001)
    elif optimizer == "Radm":
        optimizer = RAdamOptimizer(learning_rate=1e-3)
    else:
        optimizer = optimizers.Adam(lr=0.001)

    return optimizer


def build_dnn_model(input_shape, optimizer='adam'):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    # select optimizer
    optimizer = select_optimizer(optimizer)

    model.compile(optimizer=optimizer,
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model


def build_rnn_model(embedding_matrix, input_length: int, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        input_length=input_length,
        weights=[embedding_matrix],
        trainable=False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # select optimizer
    optimizer = select_optimizer(optimizer)
    model.compile(optimizer=optimizer,
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model


def build_cnn_model(vocab_size, max_len, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_len))
    model.add(Conv1D(filters=9, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # select optimizer
    optimizer = select_optimizer(optimizer)
    model.compile(optimizer=optimizer,
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model




