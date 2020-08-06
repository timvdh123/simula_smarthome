import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from discretize import Dataset
from models import single_sensor_multistep_future_encoder_decoder
from similarity import LOF_all_data, isolation_forest_all, isolation_forest, LOF
from synthesized_analysis import get_entropy_information
from training import create_train_activity_prediction_model, create_train_sensor_prediction_model, \
    synthesize_sensors_multiple_days
from utils import plot_uncertainty


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
                                                 model_name='single_step_prediction',
                                                 epochs=100,
                                                 window_size=4 * 24,
                                                 future_steps=1,
                                                 dt=900,  # Predict 30 minutes ahead.
                                                 with_time=True,
                                                 batch=128,
                                                 sensor_id=sensor_id,
                                                 features=[24, 5, 6, 9],
                                                 load_weights=False)
        except Exception as e:
            print(e)

        try:
            create_train_sensor_prediction_model(combined1,
                                                 model_args=model_args,
                                                 model_name='vector_output',
                                                 epochs=1500,
                                                 window_size=24*5,
                                                 future_steps=24,
                                                 dt=3600,  # Predict 1 day ahead.
                                                 with_time=True,
                                                 batch=128,
                                                 sensor_id=sensor_id,
                                                 features=[sensor_id],
                                                 load_weights=False)
        except Exception as e:
            print(e)
        try:
            model = single_sensor_multistep_future_encoder_decoder(
                timesteps=4*24,
                future_timesteps=24,
                n_features=3,
            )
            create_train_sensor_prediction_model(combined1,
                                                 model_args=model_args,
                                                 model_name='encoder_decoder',
                                                 model=model,
                                                 epochs=1500,
                                                 window_size=24 * 5,
                                                 future_steps=24,
                                                 dt=3600,  # Predict 30 minutes ahead.
                                                 with_time=True,
                                                 batch=128,
                                                 sensor_id=sensor_id,
                                                 features=[sensor_id],
                                                 load_weights=False)
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

def run_multiple_non_deterministic(folders, model_names, start, n_steps, output_folder, n_reps):
    for _ in tqdm(range(n_reps)):
        run_sensor_non_deterministic(folders, model_names, start, n_steps, output_folder)

if __name__ == '__main__':
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)

    combined1.sensor_data_summary()

    bathroom2 = Dataset.parse('dataset/', 'bathroom2')
    kitchen2 = Dataset.parse('dataset/', 'kitchen2')
    combined2 = bathroom2.combine(kitchen2)

    combined2.sensor_data_summary()

    # Creates LOF and isolation forest plots
    LOF(combined1, ['duration'])
    LOF(combined1, ['start_time'])
    LOF(combined1, ['duration', 'start_time'])
    LOF(combined1, ['duration', 'start_time'], 10)
    isolation_forest(combined1)

    LOF_all_data(combined1)
    isolation_forest_all(combined1)

    # Trains models for each sensor and activity separately
    run_sensor_model_training()
    run_activity_model_training()

    # Generate 10 days synthetic data
    folders = {24: 'results/model_32', 5: 'results/model_33', 6: 'results/model_30', 9: 'results/model_31'}
    model_names = {sensor_id: 'lstm_vector_output' for sensor_id in [24, 5, 6, 9]}
    start = 24 * 24 * 4
    n_steps = 24 * 4 * 10
    window_size = 24 * 4
    run_sensor_synthesis(folders, model_names, start, n_steps, window_size)
    data = pd.read_csv('output.csv', index_col=0)
    input_data = pd.read_csv('input.csv', index_col=0)

    # Print entropy information
    get_entropy_information(input_data, data)

    # Generate 10 days synthetic data 1000 times with some gaussian noise
    run_multiple_non_deterministic(folders, model_names, start, n_steps,
                                   'uncertainty_simulations', 1000)

    # Plot the results
    plot_uncertainty('uncertainty_simulations', deterministic_data=data, input_data=input_data,
                     day_length=24*4)
