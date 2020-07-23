import os

import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import metrics
import tensorflow as tf

from models import single_sensor_multistep_future


class IsNanEarlyStopper(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
          return
        if np.isnan(current):
            self.model.stop_training = True

    def on_batch_end(self, batch, logs={}):
        current = self.get_monitor_value(logs)
        if np.isnan(current):
            self.model.stop_training = True

class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        if np.isnan(logs.get('loss', 0)) or logs.get('loss', 0) > 100:
            for k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
            if self.step % self.display == 0:
                metrics_log = ''
                for (k, v) in self.metric_cache.items():
                    val = v / self.display
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
                print('step: {}/{} ... {}'.format(self.step,
                                              self.params['steps'],
                                              metrics_log))
                self.metric_cache.clear()

def prepare_data_future_steps(d, window_size = 70, dt=60,
                                     with_time=False, future_steps=20, sensor_id=24, **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    series_sensor_data = series_sensor_data[[sensor_id]]
    if with_time:
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            series_sensor_data.index.hour * 3600 + \
        series_sensor_data.index.minute *60 + \
        series_sensor_data.index.second
        series_sensor_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        series_sensor_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = np.zeros((len(series_sensor_data), future_steps, series_sensor_data.shape[1]))
    for i in range(future_steps):
        output[:, i, :] = series_sensor_data.shift(-1*i) # Future steps
    for i in range(window_size):
        #0 -> shift(window_size)
        #1 -> shift(window_size-1)
        data[:, i, :] = series_sensor_data.shift(window_size-i)
    return data[window_size:-future_steps, :, :], output[window_size:-future_steps, :]

def test_train_val_split(X, y, dt, n_val_days, n_test_days):
    X_train = X[:-1*(n_val_days+n_test_days)*(3600//dt)*24, :, :]
    X_val = X[-(n_val_days+n_test_days)*(3600//dt)*24:-n_test_days*(3600//dt)*24, :, :]
    X_test = X[-n_test_days*(3600//dt)*24:, :, :]
    y_train = y[:-1*(n_val_days+n_test_days)*(3600//dt)*24, :, :]
    y_val = y[-(n_val_days+n_test_days)*(3600//dt)*24:-n_test_days*(3600//dt)*24, :, :]
    y_test = y[-n_test_days*(3600//dt)*24:, :, :]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def train_future_timesteps(d, **kwargs):
    window_size = kwargs.setdefault('window_size', 60) #Number of steps to look back
    future_steps = kwargs.setdefault('future_steps', int(window_size*0.2)) #Number of steps to look
    # back
    epochs = kwargs.setdefault('epochs', 20)
    batch = kwargs.setdefault('batch', 128)
    dt = kwargs.setdefault('dt', 600)
    sensor_id = kwargs.setdefault('sensor_id', 24)
    X, y = prepare_data_future_steps(d, **kwargs)
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = test_train_val_split(X, y, dt, 2, 2)
    n_features = X.shape[2]  # 59
    for index, id in [(0, sensor_id)]:
        model = single_sensor_multistep_future(
            n_features=n_features,
            timesteps=window_size,
            future_timesteps=future_steps,
            learning_rate=1e-3,
            hidden_layer_activation='tanh',
            hidden_layers=1,
            hidden_layer_units=360,
            input_n_units=360,
            second_layer_input=360
        )

        y_train_sensor = y_train[:, :, index]
        y_val_sensor = y_val[:, :, index]
        y_test_sensor = y_test[:, :, index]
        print("Training data shape %s" % str(X_train.shape))
        print("Training data output shape %s" % str(y_train.shape))
        cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier_sensor_future_%d.h5" % id,
                             save_best_only=False,
                             monitor='val_accuracy',
                             verbose=0)
        if os.path.exists("lstm_autoencoder_classifier_sensor_future_%d.h5" % id):
            try:
                model.load_weights("lstm_autoencoder_classifier_sensor_future_%d.h5" % id)
            except ValueError as v:
                print("Could not load model weights")
        print("Sensor %d" % id)
        out_batch = NBatchLogger(display=1)
        early_stopper = IsNanEarlyStopper(monitor='loss')
        lstm_autoencoder_history = model.fit(X_train, y_train_sensor,
                                             epochs=epochs,
                                             batch_size=batch,
                                             verbose=1,
                                             validation_data=(X_val, y_val_sensor),
                                             callbacks=[early_stopper, out_batch, cp],
                                             shuffle=True,
                                             ).history
        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        yPredTest = model.predict(X_test)
        yPredTestBinary = (yPredTest > 0.5).astype(int)
        accuracy = metrics.binary_accuracy(y_test_sensor, yPredTestBinary)
        print("Mean Accuracy: %3.2f +- %3.2f" % (np.mean(accuracy), np.std(accuracy)))
        model.save("lstm_autoencoder_classifier_sensor_future_%d.h5" % id)

        # Remove last dimension
        yPredTest = yPredTest.reshape(yPredTest.shape[0], yPredTest.shape[1])
        yPredTestBinary = yPredTestBinary.reshape(yPredTestBinary.shape[0], yPredTestBinary.shape[1])
        y_test_sensor = y_test_sensor.reshape(y_test_sensor.shape[0], y_test_sensor.shape[1])
        print(y_test.shape)
        distances = []
        for i in range(0, y_test.shape[0], y_test.shape[0]//8):
            distance, path = fastdtw(y_test_sensor[i, :], yPredTestBinary[i, :])
            distances.append(distance)
            fig, ax = plt.subplots()
            ax.plot(y_test_sensor[i, :], label='Actual')
            ax.plot(yPredTestBinary[i, :], label='Predicted')
            plt.legend()
            plt.title("Sensor %d \n Accuracy = %3.2f\n DTW dist = %3.2f" % (
                id,
                metrics.binary_accuracy(y_test_sensor[i, :], yPredTest[i, :]),
                distance
            ))
            plt.show()
