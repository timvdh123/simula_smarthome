import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def prepare_data_single_output(d, window_size = 70, shift_direction=-1, dt=60, with_time=False,
                               **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    if with_time:
        series_sensor_data['hour'] = series_sensor_data.index.hour
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = series_sensor_data.shift(shift_direction*window_size).values
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
    lstm_autoencoder.add(
        LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(1))
    # Decoder
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
    lstm_autoencoder.add(Activation('sigmoid'))
    lstm_autoencoder.add(Dense(n_features))

    adam = Adam(lr)
    lstm_autoencoder.compile(loss='mse', optimizer=adam)
    return lstm_autoencoder

def train(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    epochs = kwargs.setdefault('epochs', 200)
    batch = kwargs.setdefault('batch', 24)
    lr = kwargs.setdefault('lr', 0.0004)
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
    cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                         verbose=0)
    lstm_autoencoder_history = model.fit(X_train, y_train, epochs=epochs,
                                                    batch_size=batch, verbose=2,
                                         validation_data=(X_validation, y_validation),
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

    yPredTest = model.predict(X_test)
    yPredTest = yPredTest[:, 0, :].reshape(X_test.shape[0], X_test.shape[2])

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