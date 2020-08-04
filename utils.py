"""File to store some utility classes

"""
import csv
import json
import os
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from tensorflow.keras.callbacks import Callback, EarlyStopping
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

def load_non_deterministic(output_folder: str):
    """Loads the results of the non-deterministic simulations

    :param output_folder: the folder where everything was saved
    :type output_folder: str
    :return: a dataframe containing all data
    :rtype:
    """
    output_csvs = \
        list(map(lambda f2: os.path.join(output_folder, f2),
                filter(lambda f: f.startswith('output'),
                   os.listdir(output_folder))))
    sum_data, original_data, count = None, None, 0
    for file in output_csvs:
        count += 1
        data = pd.read_csv(file, index_col=0)
        if sum_data is None:
            sum_data = data.values
            original_data = data
        else:
            sum_data += data.values
    sum_data/=count # Convert to probability
    return pd.DataFrame(data=sum_data, index=original_data.index, columns=original_data.columns),\
           count


def plot_uncertainty(uncertainty_simulation_output_folder,
                     deterministic_data=None,
                     input_data=None,
                     day_length=24*4):
    """Plots the results of the uncertainty simulations.

    :param uncertainty_simulation_output_folder: the folder where the simulation results were saved
    :param deterministic_data: an optional parameter used to also plot the deterministic prediction
    :param input_data: an optional parameter used to plot the daily activations of the sensors in
    the training data
    :param day_length: the length of each day
    :return: None
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    df, count = load_non_deterministic(uncertainty_simulation_output_folder)
    df.plot(ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Expected value')
    plt.title("Expected value of sensors\n"
              "Combined results of %d simulations with gaussian noise added to prediction" % count)
    plt.show()
    colors = ['C0', 'C1', 'C2', 'C3', 'm', 'y', 'k']

    if deterministic_data is not None:
        fig, axs = plt.subplots(figsize=(16, 8), nrows=len(deterministic_data.columns))
        plt.subplots_adjust(top=0.8)
        for i, col in enumerate(deterministic_data.columns):
            deterministic_data[[col]].plot(ax=axs[i], color=colors[i], label=col)
            df[[col]].plot(ax=axs[i], color=colors[i], label='%s (uncertainty)' % col,
                           linestyle='dashed')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Sensor %s' % col)
            axs[i].legend(loc='upper right')
        plt.show()

    if input_data is not None:
        vals, orig_vals, count = None, None, 0
        for i in range(0, len(input_data), day_length):
            count += 1
            if vals is None:
                vals = input_data.iloc[i:(i+day_length)].values
                orig_vals = input_data.iloc[i:(i+day_length)]
            else:
                vals += input_data.iloc[i:(i+day_length)].values
        vals /= count
        vals = pd.DataFrame(data=vals, index=orig_vals.index, columns=orig_vals.columns)

        fig, axs = plt.subplots(figsize=(10, 8), nrows=len(vals.columns))
        plt.subplots_adjust(top=0.8)
        for i, col in enumerate(vals.columns):
            x = list(map(parser.parse, vals.index.values))
            axs[i].plot(
                x,
                vals[[col]].values,
                color=colors[i],
                label=col
            )
            axs[i].xaxis.set_minor_locator(mdates.HourLocator())
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axs[i].set_xlabel('Time')
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel('Sensor %s' % col)
            axs[i].legend(loc='upper right')

        fig.show()

def combine_model_results():
    """Combines the metrics and results of multiple models and writes them to models_table.csv

    :return:
    :rtype:
    """
    paths = list(filter(lambda s: s.startswith('model_'), os.listdir('.')))

    kwargs_fields = ['model_name', 'model_number', 'sensor_id', 'window_size', 'future_steps',
                     'dt', 'with_time',
                     'batch', 'epochs']
    metrics_fields = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'mean true negatives',
                      'mean false positives', 'mean false negatives', 'mean true positives',
                      'mean_binary_accuracy']
    fieldnames = kwargs_fields + metrics_fields
    if os.path.exists('models_table.csv'):
        os.remove('models_table.csv')

    data = []
    for p in paths:
        name = os.listdir(p)[0].split('_sensor')[0]
        number = p.split('_')[1]
        kwargs, model_args, metrics = None, None, None
        kwargs_file = p + '/' + list(filter(lambda s: s.endswith('kwargs.json'), os.listdir(p)))[0]
        model_args_file = p + '/' + \
                          list(filter(lambda s: s.endswith('model_args.json'), os.listdir(p)))[0]
        metrics_file = p + '/' + list(filter(lambda s: s.endswith('metrics.json'),
                                             os.listdir(p)))[0]
        with open(kwargs_file, 'r') as f:
            kwargs = json.load(f)
        with open(model_args_file, 'r') as f:
            model_args = json.load(f)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        if model_args is not None:
            kwargs.update(model_args)
        if metrics is not None:
            kwargs.update(metrics)
        kwargs['model_name'] = name
        kwargs['model_number'] = number
        kwargs = {k: v for k, v in kwargs.items() if k in fieldnames}
        write_header = not os.path.exists('models_table.csv')
        with open('models_table.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(kwargs)
