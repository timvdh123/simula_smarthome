import csv
import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from discretize import Dataset
from training import create_train_activity_prediction_model, create_train_sensor_prediction_model, \
    synthesize_sensors_multiple_days


def combine_model_results():
    paths = list(filter(lambda s: s.startswith('model_'), os.listdir('.')))

    kwargs_fields = ['model_name', 'model_number', 'sensor_id', 'window_size', 'future_steps',
                     'dt', 'with_time',
                     'batch', 'epochs']
    metrics_fields = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'mean true negatives', 'mean false positives', 'mean false negatives', 'mean true positives', 'mean_binary_accuracy']
    fieldnames = kwargs_fields + metrics_fields
    if os.path.exists('models_table.csv'):
        os.remove('models_table.csv')

    data = []
    for p in paths:
        name = os.listdir(p)[0].split('_sensor')[0]
        number = p.split('_')[1]
        kwargs, model_args, metrics = None, None, None
        kwargs_file = p + '/' + list(filter(lambda s: s.endswith('kwargs.json'), os.listdir(p)))[0]
        model_args_file =  p + '/' + list(filter(lambda s: s.endswith('model_args.json'), os.listdir(p)))[0]
        metrics_file =  p + '/' + list(filter(lambda s: s.endswith('metrics.json'),
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

    #Run vector output model
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
                           window_size=60*24,
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
                                   window_size=4*24,
                                   future_steps=1,
                                   dt=900, # Predict 30 minutes ahead.
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
    plt.axvline(input.index[start+window_size])
    plt.title('Predicted data')
    plt.show()

    input.iloc[(start+window_size):(start+window_size + n_steps)].plot()
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

def run_multiple_non_deterministic(folders, model_names, start, n_steps, output_folder, n_reps):
    for _ in tqdm(range(n_reps)):
        run_sensor_non_deterministic(folders, model_names, start, n_steps, output_folder)

if __name__ == '__main__':
    folders = {24: 'model_32', 5: 'model_33', 6: 'model_30', 9: 'model_31'}
    model_names = {sensor_id: 'lstm_vector_output' for sensor_id in [24, 5, 6, 9]}
    start = 24*24*4
    n_steps = 24*4*10
    window_size=24*4
    # run_sensor_synthesis(folders, model_names, start, n_steps, window_size)
    run_multiple_non_deterministic(folders, model_names, start, n_steps,
                                   'uncertainty_simulations', 1000)


