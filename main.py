from discretize import Dataset
from lstm import train_activity_synthesis, train_future_timesteps, synthesize_sensors_multiple_days
import matplotlib.pyplot as plt

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

    train_activity_synthesis(combined1,
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
            train_future_timesteps(combined1,
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

if __name__ == '__main__':
    folders = {24: 'model_32', 5: 'model_33', 6: 'model_30', 9: 'model_31'}
    model_names = {sensor_id: 'lstm_vector_output' for sensor_id in [24, 5, 6, 9]}
    start = 24*24*4
    n_steps = 24*2*10
    window_size=24*4
    run_sensor_synthesis(folders, model_names, start, n_steps, window_size)

