import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from discretize import Data


def prepare_data(d: Data, window_size = 70, shift_direction=1, dt=60, with_time=True):
    series_sensor_data = d.sensor_values_reshape(dt)
    if with_time:
        series_sensor_data['hour'] = series_sensor_data.index.hour
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    for i in range(window_size):
        data[:, i, :] = series_sensor_data.shift(shift_direction*i)
    if shift_direction == 1:
        return data[window_size:, :, :]
    return data[:-window_size, :, :]

def load_model(timesteps, n_features, lr, path):
    model = get_model(timesteps, n_features, lr)
    model.load_weights(path)
    return model

def get_model(timesteps, n_features, lr):
    lstm_autoencoder = Sequential()
    lstm_autoencoder.add(
        LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(timesteps))
    # Decoder
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
    lstm_autoencoder.add(Activation('sigmoid'))

    adam = Adam(lr)
    lstm_autoencoder.compile(loss='mse', optimizer=adam)
    return lstm_autoencoder

def train(d: Data, lookback=70, epochs=200, batch=24, lr=0.0004, dt=60, shift_direction=-1,
          with_time=True):
    X = prepare_data(d, window_size=lookback, dt=dt, shift_direction=shift_direction,
                     with_time=with_time)
    X_test = X[-2*(3600//dt)*24:]
    X_validation = X[-4*(3600//dt)*24:-2*(3600//dt)*24]
    X_train = X[:-4*(3600//dt)*24]

    timesteps = lookback  # equal to the lookback
    n_features = X.shape[2]  # 59
    model = get_model(timesteps, n_features, lr)
    cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                         verbose=0)
    lstm_autoencoder_history = model.fit(X_train, X_train, epochs=epochs,
                                                    batch_size=batch, verbose=2,
                                         validation_data=X_validation,
                                                    callbacks=[cp]
                                                    ).history
    plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    X_pred = model.predict(X_train)
    X_pred = X_pred[:, 0, :].reshape(X_train.shape[0], X_train.shape[2])
    Xtrain = X_train[:, 0, :].reshape(X_train.shape[0], X_train.shape[2])

    fig, ax = plt.subplots()
    ax.set_title("Loss distribution")
    sns.distplot(np.mean(np.abs(X_pred - Xtrain), axis=1), ax=ax)
    plt.show()

    X_pred_test = model.predict(X_test)
    X_pred_test = X_pred_test[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])
    Xtest = X_test[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])

    fig, ax = plt.subplots()
    ax.set_title("Loss distribution")
    sns.distplot(np.mean(np.abs(X_pred_test - Xtest), axis=1), ax=ax)
    plt.show()

    plot_sensor_predictions(d, dt, X_pred_test, Xtest, lookback, with_time)
    return model

def plot_sensor_predictions(d, dt, X_pred, Xtest, lookback, with_time=False):
    data = d.sensor_values_reshape(dt)
    anomaly = np.mean(np.abs(X_pred - Xtest), axis=1) > 0.025

    time = np.arange(
        start=pd.Timestamp(d.sensor_data.start_time.min().date()),
        stop=pd.Timestamp(d.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
        step=pd.to_timedelta(dt, 's')
    ).astype('datetime64[ns]')
    data.index = time
    time = time[lookback:]
    X_test_time = time[-(3600//dt) * 24:]
    if with_time:
        df = pd.DataFrame(X_pred[:,:-1]-Xtest[:,:-1], columns=data.columns, index=X_test_time)
        df_pred = pd.DataFrame(X_pred[:,:-1], columns=data.columns, index=X_test_time)
        df_test = pd.DataFrame(Xtest[:,:-1], columns=data.columns, index=X_test_time)
    else:
        df = pd.DataFrame(X_pred - Xtest, columns=data.columns, index=X_test_time)
        df_pred = pd.DataFrame(X_pred, columns=data.columns, index=X_test_time)
        df_test = pd.DataFrame(Xtest, columns=data.columns, index=X_test_time)

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