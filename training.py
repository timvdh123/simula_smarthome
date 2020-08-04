import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sklearn.metrics import confusion_matrix
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from models import single_sensor_multistep_future, activity_synthesis_vector_output
from utils import NBatchLogger, IsNanEarlyStopper, train_val_test_split, prepare_data_future_steps, \
    prepare_data_activity_synthesis, save_model


def synthesize_sensors_multiple_days(dataset, folders, names, start_index, num_repetitions,
                                     deterministic=True, eps=0.2):
    """Uses a trained model to synthesize sensor future sensor values.

    :param dataset: The dataset that will be used
    :param folders: A dictionary of sensor ids -> folder names which will be used to load the models
    :param names: A dictionary of sensor ids -> model names used to load the models.
    :param start_index: Index of the dataset to start prediction
    :param num_repetitions: Number of repetitions to repeat the prediction
    :param deterministic: Whether a deterministic prediction will be used (which means that a
    predicted value >0.5 always becomes a 1, otherwise a 0).
    :param eps: When deterministic is set to False, a normal distribution with mean 0, std=eps is
    used to add noise to the predicted value
    :return: An input and output DataFrame which contain the input values and predicted values.
    """
    models, model_args_all, kwargs_all = {}, {}, {}
    kwargs = None
    for sensor_id, folder in folders.items():
        model_name = names[sensor_id]
        with open("%s/%s_sensor_%d_model_args.json" % (folder, model_name, sensor_id), 'r') as f:
            model_args = json.load(f)
        with open("%s/%s_sensor_%d_kwargs.json" % (folder, model_name, sensor_id), 'r') as f:
            kwargs = json.load(f)

        n_features = len(kwargs.get('features', [sensor_id])) + 2 # 2 time features + sensor values
        model = single_sensor_multistep_future(
            timesteps=kwargs['window_size'],
            future_timesteps=kwargs['future_steps'],
            n_features=n_features,
            **model_args)
        model.load_weights('%s/%s_sensor_%d_model.h5' % (folder, model_name, sensor_id))
        models[sensor_id] = model
        model_args_all[sensor_id] = model_args
        kwargs_all[sensor_id] = kwargs

    series_sensor_data = dataset.sensor_values_reshape(kwargs['dt'])
    s = min(start_index, len(series_sensor_data) - kwargs['window_size'])
    if s < start_index:
        num_repetitions += (start_index - s)
        start_index = s
    extra = num_repetitions + kwargs['future_steps'] + kwargs['window_size'] + start_index \
            - len(series_sensor_data)
    if extra > 0:
        new = pd.DataFrame(data=np.zeros((extra, len(series_sensor_data.columns))),
                           columns=series_sensor_data.columns)
        new.index = \
            [max(series_sensor_data.index) + pd.DateOffset(seconds=int(i)) for i in
             ((new.index + 1) * kwargs['dt']).values]
        series_sensor_data = pd.concat([series_sensor_data, new])

    features_data = series_sensor_data[kwargs['features']].copy()
    if kwargs['with_time']:
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            series_sensor_data.index.hour * 3600 + \
            series_sensor_data.index.minute * 60 + \
            series_sensor_data.index.second
        features_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        features_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)

    data = features_data.values
    data_output = np.copy(data)

    X_start = start_index
    X_end = X_start + kwargs['window_size']
    X_prediction = X_end + kwargs['future_steps']
    for _ in tqdm(range(num_repetitions)):
        if X_end >= len(data_output):
            print("End reached")
            break
        for sensor_id, model in models.items():
            d0 = data_output[X_start:X_end].reshape(1, kwargs['window_size'], data.shape[1])

            if deterministic:
                y0 = (model.predict(d0) > 0.5).astype(float)[0]
            else:
                epsilon = np.random.standard_normal()*eps
                y0 = (model.predict(d0) > (0.5 + epsilon)).astype(float)[0]
            data_output[X_end:X_prediction, kwargs['features'].index(sensor_id)] = y0

        X_start += kwargs['future_steps']
        X_end += kwargs['future_steps']
        X_prediction += kwargs['future_steps']

    output_data = pd.DataFrame(data=data_output[(start_index + kwargs['window_size']):(
            start_index + kwargs['window_size'] + num_repetitions), :len(kwargs['features'])],
                               columns=kwargs['features'],
                               index=series_sensor_data.index[(start_index + kwargs['window_size']):(
            start_index + kwargs['window_size'] + num_repetitions)])

    original_length = len(dataset.sensor_values_reshape(kwargs['dt']))
    input_data = pd.DataFrame(data=data[:original_length,:len(kwargs['features'])],
                              columns=kwargs['features'],
                              index=series_sensor_data.index[:original_length])
    return input_data, output_data

def synthesize_activity_multiple_days(dataset, folder, model_name, start_index, num_repetitions):
    """Uses a trained model to synthesize sensor future sensor values.

    :param dataset: The dataset that will be used
    :param folder: The folder where the model is stored
    :param model_name: The name of the model
    :param start_index: Index of the dataset to start prediction
    :param num_repetitions: Number of repetitions to repeat the prediction
    :return: An input and output DataFrame which contain the input values and predicted values.
    """

    model_args, kwargs = None, None
    with open("%s/%s_sensor_%d_model_args.json" % (folder, model_name, -1), 'r') as f:
        model_args = json.load(f)
    with open("%s/%s_sensor_%d_kwargs.json" % (folder, model_name, -1), 'r') as f:
        kwargs = json.load(f)
    model = activity_synthesis_vector_output(
        timesteps=kwargs['window_size'],
        future_timesteps=kwargs['future_steps'],
        n_features=3,
        **model_args)
    model.load_weights('%s/%s_sensor_%d_model.h5' % (folder, model_name, -1))

    activity_data = dataset.activity_reshape(kwargs['dt'])
    if kwargs.get('with_time', True):
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = \
            activity_data.index.hour * 3600 + \
            activity_data.index.minute * 60 + \
            activity_data.index.second
        activity_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
        activity_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)
    data = activity_data.values
    data_output = np.copy(data)

    X_start = start_index
    X_end = X_start + kwargs['window_size']
    X_prediction = X_end + kwargs['future_steps']
    for _ in range(num_repetitions):
        if X_end >= len(data_output):
            print("End reached")
            break
        d0 = data_output[X_start:X_end].reshape(1, kwargs['window_size'], 3)
        y0 = np.argmax(model.predict(d0), axis=-1)
        data_output[X_end:X_prediction, 0] = y0

        X_start += kwargs['future_steps']
        X_end += kwargs['future_steps']
        X_prediction += kwargs['future_steps']

    return data_output, data

def evaluate_sensor_model_single_timestep(
        model,
        model_name,
        sensor_id,
        X_test,
        y_test,
        training_history=None,
        save_folder=None
):
    """Evaluate model which predicts single time step ahead.
    """
    model_metrics = {}
    base_path = "%s_sensor_%d" % (model_name, sensor_id)
    if save_folder is not None:
        base_path = "%s/%s" % (save_folder, base_path)
    yPredTest = model.predict(X_test) > 0.5
    yPredTestBinary = (yPredTest > 0.5).astype(int)
    y_test = y_test[:, :, 0]
    y_test = (y_test > 0.5).astype(int)
    accuracy = metrics.binary_accuracy(y_test, yPredTestBinary)
    conf_matrix = confusion_matrix(y_test, yPredTestBinary)
    model_metrics["mean true negatives"] = np.mean(conf_matrix[0, 0])
    model_metrics["mean false positives"] = np.mean(conf_matrix[0, 1])
    model_metrics["mean false negatives"] = np.mean(conf_matrix[1, 0])
    model_metrics["mean true positives"] = np.mean(conf_matrix[1, 1])
    model_metrics["mean_binary_accuracy"] = float(np.mean(accuracy))
    base_path = "%s_sensor_%d" % (model_name, sensor_id)
    with open("%s_metrics.json" % base_path, 'w') as f:
        json.dump(model_metrics, f)
    y_test_sensor = y_test.reshape(1, -1)[0]
    yPredTestBinary = yPredTestBinary.reshape(1, -1)[0]
    distance, path = fastdtw(y_test_sensor, yPredTestBinary)
    fig, ax = plt.subplots()
    ax.plot(y_test_sensor, label='Actual')
    ax.plot(yPredTestBinary, label='Predicted')
    plt.legend()
    plt.title("Sensor %d \n Accuracy = %3.2f \n DTW distance = %3.2f" % (
        sensor_id,
        metrics.binary_accuracy(y_test.reshape(1, -1), yPredTestBinary.reshape(1, -1))[
            0].numpy(),
        distance
    ))
    plt.savefig("%s_figure.png" % base_path)
    plt.show()


def evaluate_sensor_model(
        model,
        model_name,
        sensor_id,
        X_test,
        y_test,
        training_history=None,
        save_folder=None
):
    """Evaluates a model which predicts multiple time steps ahead and writes the metrics to a
    folder.
    """
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
    if len(yPredTest.shape) == 3:
        yPredTest = yPredTest[:, 0, :]
    yPredTestBinary = (yPredTest > 0.5).astype(int)
    y_test = (y_test > 0.5).astype(int)
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

def evaluate_model_activity(model,
        model_name,
        sensor_id,
        X_test,
        y_test,
        training_history=None,
        save_folder=None
):
    """Evaluates a model and writes the metrics to a folder.
    """
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
    accuracy = (y_test == np.argmax(yPredTest, axis=-1))
    accuracy = accuracy.sum(axis=1) / accuracy.shape[1]


    print("Mean Accuracy: %3.2f +- %3.2f" % (np.mean(accuracy), np.std(accuracy)))
    model_metrics["mean_binary_accuracy"] = float(np.mean(accuracy))

    print(y_test.shape)
    distances = []

    yPredTest = np.argmax(yPredTest, axis=-1)
    for i in range(0, y_test.shape[0], y_test.shape[0] // 8):
        distance, path = fastdtw(y_test[i, :], yPredTest[i, :])
        distances.append(distance)
        fig, ax = plt.subplots()
        ax.plot(y_test[i, :], label='Actual')
        ax.plot(yPredTest[i, :], label='Predicted')
        plt.legend()
        plt.title("Sensor %d \n Accuracy = %3.2f\n DTW dist = %3.2f" % (
            sensor_id,
            np.mean(y_test[i, :] == yPredTest[i, :]),
            distance
        ))
        plt.savefig("%s_figure_%d.png" % (base_path, i))
        plt.show()

    with open("%s_metrics.json" % base_path, 'w') as f:
        json.dump(model_metrics, f)

def create_train_activity_prediction_model(
        d,
        model_name,
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
    X, y = prepare_data_activity_synthesis(d, **kwargs)
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = train_val_test_split(X, y, dt, 2, 2)
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
            model = activity_synthesis_vector_output(timesteps=window_size,
                                                   future_timesteps=future_steps,
                                                   n_features=n_features,
                                                   **model_args)

        if len(y.shape) == 3:
            y_train_sensor = y_train[:, :, index]
            y_val_sensor = y_val[:, :, index]
            y_test_sensor = y_test[:, :, index]
        else:
            y_train_sensor = y_train
            y_val_sensor = y_val
            y_test_sensor = y_test

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
        evaluate_model_activity(model, model_name, sensor_id, X_test, y_test, model_history,
                                folder)

def create_train_sensor_prediction_model(
        d,
        model_name,
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
    ) = train_val_test_split(X, y, dt, 2, 2)
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
        if future_steps == 1:
            evaluate_sensor_model_single_timestep(model, model_name, sensor_id, X_test,
                                                  y_test_sensor, training_history=model_history,
                                                  save_folder=folder)
        else:
            evaluate_sensor_model(model, model_name, sensor_id, X_test, y_test_sensor,
                                  training_history=model_history, save_folder=folder)

def run_activity_model_training():
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)

    # Run vector output model
    model_args = {
        "learning_rate": 1e-3,
        "hidden_layer_activation": 'tanh',
        "hidden_layers": 1,
        "hidden_layer_units": 120,
        "input_n_units": 120,
        "second_layer_input": 120,
        "n_activities": 25
    }

    create_train_activity_prediction_model(combined1,
                                           model_args=model_args,
                                           model_name='vector_output_activity_synthesis',
                                           epochs=100,
                                           window_size=60 * 24,
                                           future_steps=1,
                                           dt=60,
                                           with_time=True,
                                           batch=128,
                                           load_weights=True,
                                           sensor_id=-1)
