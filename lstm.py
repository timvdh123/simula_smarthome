import json
import os

import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.vis_utils import plot_model

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


def prepare_data_future_steps(d, window_size=70, dt=60,
                              with_time=False, future_steps=20, sensor_id=24, **kwargs):
    series_sensor_data = d.sensor_values_reshape(dt)
    series_sensor_data = series_sensor_data[[sensor_id]]
    if with_time:
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            series_sensor_data.index.hour * 3600 + \
            series_sensor_data.index.minute * 60 + \
            series_sensor_data.index.second
        series_sensor_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        series_sensor_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
    data = np.zeros((len(series_sensor_data), window_size, series_sensor_data.shape[1]))
    output = np.zeros((len(series_sensor_data), future_steps, series_sensor_data.shape[1]))
    for i in range(future_steps):
        output[:, i, :] = series_sensor_data.shift(-1 * i)  # Future steps
    for i in range(window_size):
        # 0 -> shift(window_size)
        # 1 -> shift(window_size-1)
        data[:, i, :] = series_sensor_data.shift(window_size - i)
    return data[window_size:-future_steps, :, :], output[window_size:-future_steps, :]


def test_train_val_split(X, y, dt, n_val_days, n_test_days):
    X_train = X[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24, :, :]
    X_val = X[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24, :,
            :]
    X_test = X[-n_test_days * (3600 // dt) * 24:, :, :]
    y_train = y[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24, :, :]
    y_val = y[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24, :,
            :]
    y_test = y[-n_test_days * (3600 // dt) * 24:, :, :]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def evaluate_model(
        model,
        model_name,
        sensor_id,
        X_test,
        y_test,
        training_history=None,
        save_folder=None
):
    model_metrics = {}
    base_path = "%s_sensor_%d" % (model_name, sensor_id)
    if save_folder is not None:
        base_path = "%s/%s" % (save_folder, base_path)
    if training_history is not None:
        plt.plot(training_history['loss'], linewidth=2, label='Train')
        plt.plot(training_history['val_loss'], linewidth=2, label='Validation')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("%s_loss.png" % base_path)
        plt.show()

        plt.plot(training_history['accuracy'], linewidth=2, label='Train')
        plt.plot(training_history['val_accuracy'], linewidth=2, label='Validation')
        plt.legend(loc='upper right')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig("%s_accuracy.png" % base_path)
        plt.show()
        model_metrics = {
            'loss': training_history['loss'][-1],
            'accuracy': training_history['accuracy'][-1],
            'val_loss': training_history['val_loss'][-1],
            'val_accuracy': training_history['val_accuracy'][-1],
        }
    yPredTest = model.predict(X_test)
    yPredTestBinary = (yPredTest > 0.5).astype(int)
    accuracy = metrics.binary_accuracy(y_test, yPredTestBinary)
    conf_matrix = np.array([confusion_matrix(y_test[i], yPredTestBinary[i])
                            for i in range(len(yPredTestBinary))])
    model_metrics["mean true negatives"] = np.mean(conf_matrix[:, 0, 0])
    model_metrics["mean false positives"] = np.mean(conf_matrix[:, 0, 1])
    model_metrics["mean false negatives"] = np.mean(conf_matrix[:, 1, 0])
    model_metrics["mean true positives"] = np.mean(conf_matrix[:, 1, 1])

    print("Mean Accuracy: %3.2f +- %3.2f" % (np.mean(accuracy), np.std(accuracy)))
    model_metrics["mean_binary_accuracy"] = float(np.mean(accuracy))

    # Remove last dimension
    yPredTest = yPredTest.reshape(yPredTest.shape[0], yPredTest.shape[1])
    yPredTestBinary = yPredTestBinary.reshape(yPredTestBinary.shape[0], yPredTestBinary.shape[1])
    y_test_sensor = y_test.reshape(y_test.shape[0], y_test.shape[1])
    print(y_test.shape)
    distances = []
    for i in range(0, y_test.shape[0], y_test.shape[0] // 8):
        distance, path = fastdtw(y_test_sensor[i, :], yPredTestBinary[i, :])
        distances.append(distance)
        fig, ax = plt.subplots()
        ax.plot(y_test_sensor[i, :], label='Actual')
        ax.plot(yPredTestBinary[i, :], label='Predicted')
        plt.legend()
        plt.title("Sensor %d \n Accuracy = %3.2f\n DTW dist = %3.2f" % (
            sensor_id,
            metrics.binary_accuracy(y_test_sensor[i, :], yPredTest[i, :]),
            distance
        ))
        plt.savefig("%s_figure_%d.png" % (base_path, i))
        plt.show()

    with open("%s_metrics.json" % base_path, 'w') as f:
        json.dump(model_metrics, f)


def save_model(model_name, model, model_args, kwargs):
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


def load_model_parameters(folder, model_name, sensor_id):
    model_args, kwargs = None, None
    with open("%s/%s_sensor_%d_model_args.json" % (folder, model_name, sensor_id), 'r') as f:
        model_args = json.load(f)
    with open("%s/%s_sensor_%d_kwargs.json" % (folder, model_name, sensor_id), 'w') as f:
        kwargs = json.load(f)
    model = single_sensor_multistep_future(**model_args)
    model.load_weights('%s/%s_sensor_%d_model.h5' % (folder, model_name, sensor_id))
    return model_args, kwargs, model


def train_future_timesteps(d, model_name,
                           load_weights=False,
                           model=None,
                           model_args=None,
                           **kwargs):
    window_size = kwargs.setdefault('window_size', 60)  # Number of steps to look back
    future_steps = kwargs.setdefault('future_steps',
                                     int(window_size * 0.2))  # Number of steps to look
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
        if model is None:
            if model_args is None:
                model_args = {
                    "learning_rate": 1e-3,
                    "hidden_layer_activation": 'tanh',
                    "hidden_layers": 1,
                    "hidden_layer_units": 120,
                    "input_n_units": 120,
                    "second_layer_input": 120
                }
            model = single_sensor_multistep_future(timesteps=window_size,
                                                   future_timesteps=future_steps,
                                                   n_features=n_features,
                                                   **model_args)

        y_train_sensor = y_train[:, :, index]
        y_val_sensor = y_val[:, :, index]
        y_test_sensor = y_test[:, :, index]
        print("Training data shape %s" % str(X_train.shape))
        print("Training data output shape %s" % str(y_train.shape))
        model_filepath = "%s_%d.h5" % (model_name, id)
        cp = ModelCheckpoint(filepath=model_filepath,
                             save_best_only=False,
                             monitor='val_accuracy',
                             verbose=0)
        cp_best = ModelCheckpoint(filepath="%s_%d_best.h5" % (model_name, id),
                             save_best_only=True,
                             monitor='val_accuracy',
                             verbose=0)
        if load_weights:
            if os.path.exists(model_filepath):
                try:
                    model.load_weights(model_filepath)
                except ValueError as v:
                    print("Could not load model weights")
        print("Sensor %d" % id)
        out_batch = NBatchLogger(display=1)
        early_stopper = IsNanEarlyStopper(monitor='loss')
        model_history = model.fit(X_train, y_train_sensor,
                                  epochs=epochs,
                                  batch_size=batch,
                                  verbose=1,
                                  validation_data=(X_val, y_val_sensor),
                                  callbacks=[early_stopper, out_batch, cp, cp_best],
                                  shuffle=True,
                                  ).history
        model.save(model_filepath)
        folder = save_model(model_name, model, model_args, kwargs)
        evaluate_model(model, model_name, sensor_id, X_test, y_test_sensor, model_history,
                       save_folder=folder)
