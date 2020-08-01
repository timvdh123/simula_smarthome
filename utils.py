"""File to store some utility classes

"""
import json
import os
from typing import Callable

from tensorflow.keras.callbacks import Callback, EarlyStopping

import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model

from discretize import Dataset


class IsNanEarlyStopper(EarlyStopping):
    """Stops training when a NAN loss/accuracy occurs

    """
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
        super().__init__()
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

def prepare_data_future_steps(d: Dataset,
                              window_size=70,
                              dt=60,
                              with_time=False,
                              future_steps=20,
                              sensor_id=24,
                              features=[24, 6, 5],
                              **kwargs):
    """Prepares sensor data from a dataset for prediction using an LSTM/convolutional network.
    The sensor data is first split into time windows of length dt. All sensors are discarded
    except for those in the features array. A window of past values of size window_size is used
    to predict future_steps ahead for a specific sensor (the sensor with id sensor_id).

    :param with_time: If set to true, time is encoded into a sinusoid with period 24 hours.
    :param sensor_id: The sensor that will be predicted. This is used in the output vector
    :param window_size: The number of past values that will be used
    :param future_steps: The number of future values that will be predicted
    :returns: two numpy arrays, X (input) and y. X has shape [#Samples, #Future steps, #Features]
    and y has shape [#Samples, #Future steps]
    """
    series_sensor_data = d.sensor_values_reshape(dt)
    features_data = series_sensor_data[features]
    series_sensor_data = series_sensor_data[[sensor_id]]
    if with_time:
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            series_sensor_data.index.hour * 3600 + \
            series_sensor_data.index.minute * 60 + \
            series_sensor_data.index.second
        series_sensor_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        series_sensor_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
        features_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        features_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
    data = np.zeros((len(series_sensor_data), window_size, features_data.shape[1]))
    output = np.zeros((len(series_sensor_data), future_steps, series_sensor_data.shape[1]))
    for i in range(future_steps):
        output[:, i, :] = series_sensor_data.shift(-1 * i)  # Future steps
    for i in range(window_size):
        # 0 -> shift(window_size)
        # 1 -> shift(window_size-1)
        data[:, i, :] = features_data.shift(window_size - i)
    return data[window_size:-future_steps, :, :], output[window_size:-future_steps, :]

def prepare_data_activity_synthesis(d, window_size=70, dt=60,
                              with_time=False, future_steps=20, **kwargs):
    """Same as prepare_data_future_steps, except for activities instead of sensors.
    """
    series_sensor_data = d.activity_reshape(dt)
    if with_time:
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            series_sensor_data.index.hour * 3600 + \
            series_sensor_data.index.minute * 60 + \
            series_sensor_data.index.second
        series_sensor_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        series_sensor_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = np.zeros((len(series_sensor_data), future_steps))
    for i in range(future_steps):
        output[:, i] = series_sensor_data.shift(-1 * i)[['activity']].values.reshape(1, -1)  # Future steps
    for i in range(window_size):
        # 0 -> shift(window_size)
        # 1 -> shift(window_size-1)
        data[:, i, :] = series_sensor_data.shift(window_size - i)
    return data[window_size:-future_steps, :, :], output[window_size:-future_steps, :]


def train_val_test_split(X: np.array, y: np.array, dt: int, n_val_days: int, n_test_days: int):
    """Splits X, y into a train, validation and test set.
    """
    X_train = X[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24]
    X_val = X[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24]
    X_test = X[-n_test_days * (3600 // dt) * 24:]
    y_train = y[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24]
    y_val = y[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24]
    y_test = y[-n_test_days * (3600 // dt) * 24:]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

def save_model(model_name, model, model_args, kwargs):
    """Saves a model to a folder. Also saves the configuration files necessary to load the model.
    """
    paths = list(
        map(lambda f: int(f[6:]), filter(lambda s: s.startswith('model_'), os.listdir('.'))))
    if len(paths) == 0:
        folder = 'model_0'
    else:
        folder = 'model_%d' % (max(paths) + 1)
    os.mkdir(folder)
    model.save('%s/%s_sensor_%d_model.h5' % (folder, model_name, kwargs['sensor_id']))
    plot_model(model, '%s/%s_sensor_%d_model.png' % (folder, model_name, kwargs['sensor_id']),
               show_shapes=True)
    with open("%s/%s_sensor_%d_model_args.json" % (folder, model_name, kwargs['sensor_id']),
              'w') as f:
        json.dump(model_args, f)
    with open("%s/%s_sensor_%d_kwargs.json" % (folder, model_name, kwargs['sensor_id']), 'w') as f:
        json.dump(kwargs, f)
    return folder


def load_model_parameters(
        folder: str,
        model_name: str,
        sensor_id: int,
        model_creation_function: Callable):
    """Loads a model from the folder where it was saved.

    :param folder: the folder where the model is saved
    :param model_name: the name of the model
    :param sensor_id: the id of the sensor
    :param model_creation_function: the function to which model_args will be passed,
    used to create the model

    :return: a tuple of model args, kwargs, model
    """
    model_args, kwargs = None, None
    with open("%s/%s_sensor_%d_model_args.json" % (folder, model_name, sensor_id), 'r') as f:
        model_args = json.load(f)
    with open("%s/%s_sensor_%d_kwargs.json" % (folder, model_name, sensor_id), 'r') as f:
        kwargs = json.load(f)
    model = model_creation_function(**model_args)
    model.load_weights('%s/%s_sensor_%d_model.h5' % (folder, model_name, sensor_id))
    return model_args, kwargs, model

