import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model

def prepare_data_future_steps(d, window_size = 70, dt=60,
                                     with_time=False, future_steps=20, **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    if with_time:
        series_sensor_data['hour'] = series_sensor_data.index.hour
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = np.zeros((len(series_sensor_data), future_steps, series_sensor_data.shape[1]))
    for i in range(future_steps):
        output[:, i, :] = series_sensor_data.shift(i) # Future steps
    for i in range(window_size):
        data[:, i, :] = series_sensor_data.shift(-1*i)
    return data[future_steps:-window_size, :, :], output[future_steps:-window_size, :]


def prepare_data_single_output(d, window_size = 70, shift_direction=-1, dt=60, with_time=False,
                               **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    if with_time:
        series_sensor_data['hour'] = series_sensor_data.index.hour
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = series_sensor_data.loc[:, series_sensor_data.columns != 'hour'].shift(shift_direction*window_size).values
    for i in range(window_size):
        data[:, i, :] = series_sensor_data.shift(shift_direction*i)
    if shift_direction == 1:
        return data[window_size:, :, :]
    return data[:-window_size, :, :], output[:-window_size, :]

def prepare_data(d, window_size = 70, shift_direction=-1, dt=60, with_time=False, **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    if with_time:
        series_sensor_data['hour'] = series_sensor_data.index.hour
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    for i in range(window_size):
        data[:, i, :] = series_sensor_data.shift(shift_direction*i)
    if shift_direction == 1:
        return data[window_size:, :, :]
    return data[:-window_size, :, :], data[:-window_size, :, :]

def load_model(timesteps, n_features, lr, path):
    model = get_model(timesteps, n_features, lr)
    model.load_weights(path)
    return model

def get_model(timesteps, n_features, lr):
    lstm_autoencoder = Sequential()
    lstm_autoencoder.add(LSTM(10, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_autoencoder.add(LSTM(6, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(1, activation='relu'))
    lstm_autoencoder.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_autoencoder.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_autoencoder.add(Dense(n_features, activation='sigmoid'))
    adam = Adam(lr)
    lstm_autoencoder.compile(loss='binary_crossentropy', optimizer=adam)
    lstm_autoencoder.summary()
    return lstm_autoencoder

def get_model_individual_sensor(timesteps, n_features, lr):
    lstm_regular = Sequential()
    lstm_regular.add(LSTM(15, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_regular.add(LSTM(8, activation='relu', return_sequences=False))
    lstm_regular.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr)
    lstm_regular.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    lstm_regular.summary()
    return lstm_regular

def get_model_all_sensors(timesteps, n_features, lr):
    lstm_regular = Sequential()
    lstm_regular.add(LSTM(15, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_regular.add(Dropout(0.3))
    lstm_regular.add(LSTM(8, activation='relu', return_sequences=False))
    lstm_regular.add(Dropout(0.3))
    lstm_regular.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(Dropout(0.3))
    lstm_regular.add(Dense(10, kernel_initializer='glorot_normal', activation='relu',
                           kernel_regularizer=l1(1e-4)))
    lstm_regular.add(Dropout(0.3))
    lstm_regular.add(Dense(n_features, activation='sigmoid'))
    adam = Adam(lr)
    lstm_regular.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    lstm_regular.summary()
    plot_model(lstm_regular, 'model.png', show_shapes=True)
    return lstm_regular

def get_model_future_predictions_sensors(timesteps, future_timesteps, n_features, lr):
    model = Sequential()
    # encoder
    model.add(LSTM(15, activation='relu', input_shape=(timesteps, n_features),
                   return_sequences=False))
    # model.add(LSTM(8, activation='relu', return_sequences=False))
    model.add(RepeatVector(future_timesteps))
    #decoder
    model.add(LSTM(15, activation='relu', return_sequences=True))
    # model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    adam = Adam(lr)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    plot_model(model, 'model.png', show_shapes=True)
    return model

def train_future_timesteps(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    future_steps = kwargs.setdefault('future_steps', int(window_size*0.2)) #Number of steps to look
    # back
    epochs = kwargs.setdefault('epochs', 20)
    batch = kwargs.setdefault('batch', 24)
    lr = kwargs.setdefault('lr', 0.0001)
    dt = kwargs.setdefault('dt', 600)
    X, y = prepare_data_future_steps(d, **kwargs)
    X_train = X[:-2*(3600//dt)*24]
    X_test = X[-2*(3600//dt)*24:]
    n_features = X.shape[2]  # 59
    for index, id in enumerate(d.sensor_data.id.unique()):
        model = get_model_future_predictions_sensors(n_features=n_features,
                                                     timesteps=window_size, lr=lr,
                                                     future_timesteps=future_steps)
        y_train = y[:-2 * (3600 // dt) * 24, :, index]
        y_test = y[-2 * (3600 // dt) * 24:, :, index]
        print("Training data shape %s" % str(X_train.shape))
        print("Training data output shape %s" % str(y_train.shape))
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
        cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier_sensor_future_%d.h5" % id,
                             verbose=0)
        if os.path.exists("lstm_autoencoder_classifier_sensor_future_%d.h5" % id):
            try:
                model.load_weights("lstm_autoencoder_classifier_sensor_future_%d.h5" % id)
            except ValueError as v:
                print("Could not load model weights")
        print("Sensor %d" % id)
        lstm_autoencoder_history = model.fit(X_train, y_train,
                                             epochs=epochs,
                                             batch_size=batch,
                                             verbose=2,
                                             callbacks=[cp],
                                             ).history
        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        yPredTest = model.predict(X_test)
        yPredTest = (yPredTest > 0.5).astype(int)

        # Remove last dimension
        yPredTest = yPredTest.reshape(yPredTest.shape[0], yPredTest.shape[1])
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
        print(yPredTest.shape)
        # print(metrics.classification_report(y_test, yPredTest))
        for i in range(y_test.shape[0]):
            fig, ax = plt.subplots()
            ax.plot(y_test[i,:], label='Actual')
            ax.plot(yPredTest[i,:], label='Predicted')
            plt.legend()
            plt.title("Sensor %d" % id)
            plt.show()


def train_parallel_sensors(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    epochs = kwargs.setdefault('epochs', 20)
    batch = kwargs.setdefault('batch', 24)
    lr = kwargs.setdefault('lr', 0.0001)
    dt = kwargs.setdefault('dt', 600)
    X, y = prepare_data_single_output(d, **kwargs)
    X_train = X[:-4*(3600//dt)*24]
    X_test = X[-4*(3600//dt)*24:]
    if kwargs.get('with_time', False):
        y_train = y[:-4 * (3600 // dt) * 24, :-1]
        y_test = y[-4 * (3600 // dt) * 24:, :-1]
    else:
        y_train = y[:-4 * (3600 // dt) * 24, :]
        y_test = y[-4 * (3600 // dt) * 24:, :]
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))
    print("Training data shape: %s" % str(X_train.shape))
    n_features = X.shape[2]  # 59
    model = get_model_all_sensors(n_features=n_features, timesteps=window_size, lr=lr)
    cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier_sensor_all_sensors.h5",
                         verbose=0)
    lstm_autoencoder_history = model.fit(X_train, y_train,
                                         epochs=epochs,
                                         batch_size=batch,
                                         verbose=2,
                                         callbacks=[cp],
                                         ).history
    plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    yPredTest = model.predict(X_test)
    if len(yPredTest.shape) == 3:
        yPredTest = yPredTest[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])
    yPredTest = (yPredTest > 0.5).astype(int)
    for index, id in enumerate(d.sensor_data.id.unique()):
        print(metrics.classification_report(y_test[:,index], yPredTest[:, index]))
        fig, ax = plt.subplots()
        ax.plot(y_test[:,index], label='Actual')
        ax.plot(yPredTest[:, index], label='Predicted')
        ax.legend()
        plt.title("Sensor %d" % id)
        plt.show()


def train_every_sensor(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    epochs = kwargs.setdefault('epochs', 20)
    batch = kwargs.setdefault('batch', 24)
    lr = kwargs.setdefault('lr', 0.0004)
    dt = kwargs.setdefault('dt', 600)


    X, y = prepare_data_single_output(d, **kwargs)
    X_test = X[-2*(3600//dt)*24:]
    X_validation = X[-4*(3600//dt)*24:-2*(3600//dt)*24]
    X_train = X[:-4*(3600//dt)*24]

    n_features = X.shape[2]  # 59
    for index, id in enumerate(d.sensor_data.id.unique()):
        model = get_model_individual_sensor(n_features=n_features, timesteps=window_size, lr=lr)
        print("Training data shape %s" % str(X_train.shape))
        y_train = y[:-4 * (3600 // dt) * 24, index]
        y_test = y[-2 * (3600 // dt) * 24:, index]
        y_validation = y[-4 * (3600 // dt) * 24:-2 * (3600 // dt) * 24, index]

        cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier_sensor_%d.h5" % id,
                             verbose=0)
        print("Sensor %d" % id)
        weight = class_weight.compute_class_weight('balanced', classes=np.unique(y_train),
                                                   y=y_train)
        weight /= sum(weight)
        weight = dict(enumerate(weight))
        print(weight)
        if weight[1] < 0.8:
            weight = None
        lstm_autoencoder_history = model.fit(X_train, y_train,
                                             epochs=epochs,
                                             batch_size=batch,
                                             verbose=2,
                                             validation_data=(X_validation, y_validation),
                                             callbacks=[cp],
                                             class_weight=weight
                                             ).history
        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        yPredTest = model.predict(X_test)
        if len(yPredTest.shape) == 3:
            yPredTest = yPredTest[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])
        yPredTest = (yPredTest > 0.5).astype(int)
        print(metrics.classification_report(y_test, yPredTest))
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Actual')
        ax.plot(yPredTest, label='Predicted')
        plt.title("Sensor %d" % id)
        plt.show()


def train(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    epochs = kwargs.setdefault('epochs', 20)
    batch = kwargs.setdefault('batch', 24)
    lr = kwargs.setdefault('lr', 0.004)
    dt = kwargs.setdefault('dt', 600)


    X, y = prepare_data_single_output(d, **kwargs)
    X_test = X[-2*(3600//dt)*24:]
    y_test = y[-2*(3600//dt)*24:]
    X_validation = X[-4*(3600//dt)*24:-2*(3600//dt)*24]
    y_validation = y[-4*(3600//dt)*24:-2*(3600//dt)*24]
    X_train = X[:-4*(3600//dt)*24]
    y_train = y[:-4*(3600//dt)*24]


    n_features = X.shape[2]  # 59
    model = get_model(n_features=n_features, timesteps=window_size, lr=lr)
    print("Training data shape %s" % str(X_train.shape))

    cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                         verbose=0)
    lstm_autoencoder_history = model.fit(X_train, y_train,
                                         epochs=epochs,
                                         batch_size=batch,
                                         verbose=2,
                                         validation_data=(X_validation, y_validation),
                                         callbacks=[cp],
                                         ).history
    plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    yPredTrain = model.predict(X_train)
    if len(yPredTrain.shape) == 3:
        yPredTrain = yPredTrain[:, 0, :].reshape(X_train.shape[0], X_train.shape[2])

    fig, ax = plt.subplots()
    ax.set_title("Loss distribution")
    sns.distplot(np.mean(np.abs(yPredTrain - y_train), axis=1), ax=ax)
    plt.show()

    yPredTest = model.predict(X_test)
    if len(yPredTest.shape) == 3:
        yPredTest = yPredTest[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])
    yPredTest = (yPredTest > 0.5).astype(int)
    print(metrics.classification_report(y_test, yPredTest, target_names=list(
        map(lambda s: 'Sensor %s' % s, d.sensor_data.id.unique()))))

    fig, ax = plt.subplots()
    ax.set_title("Loss distribution")
    sns.distplot(np.mean(np.abs(yPredTest - y_test), axis=1), ax=ax)
    plt.show()

    plot_sensor_predictions(d, dt, yPredTest, y_test, window_size, False)
    return model

def plot_sensor_predictions(d, dt, yPred, yTest, lookback, with_time=False):
    print(yPred.shape)
    data = d.sensor_values_reshape(dt)
    anomaly = np.mean(np.abs(yPred - yTest), axis=1) > 0.025

    time = np.arange(
        start=pd.Timestamp(d.sensor_data.start_time.min().date()),
        stop=pd.Timestamp(d.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
        step=pd.to_timedelta(dt, 's')
    ).astype('datetime64[ns]')
    data.index = time
    time = time[lookback:]
    X_test_time = time[-2 * (3600//dt) * 24:]
    print(X_test_time.shape)
    if with_time:
        df = pd.DataFrame(yPred[:, :-1] - yTest[:, :-1], columns=data.columns, index=X_test_time)
        df_pred = pd.DataFrame(yPred[:, :-1], columns=data.columns, index=X_test_time)
        df_test = pd.DataFrame(yTest[:, :-1], columns=data.columns, index=X_test_time)
    else:
        df = pd.DataFrame(yPred - yTest, columns=data.columns, index=X_test_time)
        df_pred = pd.DataFrame(yPred, columns=data.columns, index=X_test_time)
        df_test = pd.DataFrame(yTest, columns=data.columns, index=X_test_time)

    # print(X_test_time[anomaly])
    series_sensor_data = d.sensor_values_reshape(600)
    # series_sensor_data.head(1000)
    # print(df.head(1000))
    # df.plot()
    # plt.show()
    # data.head(24 * 60).plot()
    # plt.show()
    for i in df.columns:
        d2 = pd.DataFrame()
        d2['predicted'] = df_pred[i]
        d2['actual'] = df_test[i]
        d2.plot()
        plt.legend()
        plt.title("Sensor %d" % i)
        plt.show()