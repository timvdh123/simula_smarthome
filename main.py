import csv
import json
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm
from dateutil import parser

from discretize import Dataset
from training import create_train_activity_prediction_model, create_train_sensor_prediction_model, \
    synthesize_sensors_multiple_days


def combine_model_results():
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


def run_sensor_model_training():
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)

    for sensor_id in [24, 5, 6, 9]:
        model_args = {
            "learning_rate": 1e-3,
            "hidden_layer_activation": 'tanh',
            "hidden_layers": 1,
            "hidden_layer_units": 120,
            "input_n_units": 120,
            "second_layer_input": 120
        }
        try:
            create_train_sensor_prediction_model(combined1,
                                                 model_args=model_args,
                                                 model_name='lstm_vector_output',
                                                 epochs=100,
                                                 window_size=4 * 24,
                                                 future_steps=1,
                                                 dt=900,  # Predict 30 minutes ahead.
                                                 with_time=True,
                                                 batch=128,
                                                 sensor_id=sensor_id,
                                                 features=[24, 5, 6, 9],
                                                 load_weights=True)
        except Exception as e:
            print(e)


def run_sensor_synthesis(folders, model_names, start, n_steps, window_size):
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)
    input, output = synthesize_sensors_multiple_days(combined1, folders, model_names, start,
                                                     n_steps)

    input.to_csv('input.csv')
    output.to_csv('output.csv')
    output.plot()
    plt.legend()
    plt.axvline(input.index[start + window_size])
    plt.title('Predicted data')
    plt.show()

    input.iloc[(start + window_size):(start + window_size + n_steps)].plot()
    plt.legend()
    plt.title('Input data')
    plt.show()


def run_sensor_non_deterministic(folders, model_names, start, n_steps, output_folder):
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)
    input, output = synthesize_sensors_multiple_days(combined1, folders, model_names, start,
                                                     n_steps, deterministic=False)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    numbers = list(map(lambda f2: int(f2.split('_')[1].split('.csv')[0]),
                       filter(lambda f: f.startswith('output'),
                              os.listdir(output_folder))))
    if len(numbers) == 0:
        next_number = 0
    else:
        next_number = max(numbers) + 1
    output.to_csv(os.path.join(output_folder, 'output_%d.csv' % next_number))
    input.to_csv(os.path.join(output_folder, 'input_%d.csv' % next_number))

def load_non_deterministic(output_folder):
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
    fig, ax = plt.subplots(figsize=(16, 8))
    df, count = load_non_deterministic(uncertainty_simulation_output_folder)
    df.plot(ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Expected value')
    plt.title("Expected value of sensors\n"
              "Combined results of %d simulations with gaussian noise added to prediction" % count)
    plt.show()

    if deterministic_data is not None:
        colors = ['C0', 'C1', 'C2', 'C3', 'm', 'y', 'k']
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

def run_multiple_non_deterministic(folders, model_names, start, n_steps, output_folder, n_reps):
    for _ in tqdm(range(n_reps)):
        run_sensor_non_deterministic(folders, model_names, start, n_steps, output_folder)

if __name__ == '__main__':
    data = pd.read_csv('output.csv', index_col=0)
    input_data = pd.read_csv('input.csv', index_col=0)
    plot_uncertainty('uncertainty_simulations', deterministic_data=data, input_data=input_data,
                     day_length=24*4)
    # folders = {24: 'model_32', 5: 'model_33', 6: 'model_30', 9: 'model_31'}
    # model_names = {sensor_id: 'lstm_vector_output' for sensor_id in [24, 5, 6, 9]}
    # start = 24 * 24 * 4
    # n_steps = 24 * 4 * 10
    # window_size = 24 * 4
    # run_sensor_synthesis(folders, model_names, start, n_steps, window_size)
    # run_multiple_non_deterministic(folders, model_names, start, n_steps,
    #                                'uncertainty_simulations', 1000)
