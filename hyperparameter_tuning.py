from kerastuner import HyperModel, RandomSearch
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from discretize import Dataset
from lstm import prepare_data_future_steps, IsNanEarlyStopper
import numpy as np

class FuturePredictionModelHyperparameters(HyperModel):
    def __init__(self, window_size, num_features, future_steps):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.future_steps = future_steps

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.LSTM(units=hp.Int('units_lstm', 5, 20, 5),
                              activation=hp.Choice('act_input', ['relu', 'tanh']),
                              input_shape=(self.window_size, self.num_features),
                       return_sequences=False))
        if hp.Boolean('extra_layer_1'):
            model.add(layers.Dense(units=hp.Int('units_hidden_%d' % 1, 5, 20, 5,
                                                parent_name='extra_layer_1',
                                                parent_values=[True]),
                                  activation=hp.Choice('act_1', ['relu', 'tanh'],
                                                       parent_name='extra_layer_1', parent_values=[True])))
        model.add(layers.Dense(self.future_steps,
                               activation=hp.Choice('act_output', ['sigmoid', 'softmax'])))
        adam = optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

def search(dt=600,
           window_size = 360,
           future_steps = 144,
           epochs = 50,
           with_time = True,
           batch_size = 128,
           max_trials = 200):
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)


    X, y = prepare_data_future_steps(combined1, window_size=window_size, dt=dt, with_time=with_time,
                                     future_steps=future_steps)
    X_train = X[:-4*(3600//dt)*24, :, :]
    X_val = X[-4*(3600//dt)*24:-2*(3600//dt)*24, :, :]
    X_test = X[-2*(3600//dt)*24:, :, :]

    # For now only sensor 24
    y_train = y[:-4 * (3600 // dt) * 24, :, 0]
    y_val = y[-4 * (3600 // dt) * 24: -2 * (3600 // dt) * 24, :, 0]
    y_test = y[-2 * (3600 // dt) * 24:, :, 0]


    tuner = RandomSearch(
        FuturePredictionModelHyperparameters(window_size=window_size, num_features=X.shape[2], future_steps=future_steps),
        objective='val_loss',
        max_trials=max_trials,
        directory='test_dir')

    tuner.search_space_summary()

    tuner.search(x=X_train,
                 y=y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 callbacks=[
                     IsNanEarlyStopper(monitor='loss')
                 ])

    tuner.results_summary()

def load_results():
    tuner = RandomSearch(
        FuturePredictionModelHyperparameters(window_size=720, num_features=5,
                                             future_steps=144),
        objective='val_loss',
        max_trials=100,
        directory='test_dir')
    tuner.reload()
    best = tuner.oracle.get_best_trials(20)
    best.sort(key = lambda trial: trial.metrics.get_best_value('val_accuracy')
    if not np.isnan(trial.metrics.get_best_value('val_loss')) else np.nan, reverse=True)

    best_accuracy = list(map(lambda trial: trial.metrics.get_best_value('val_accuracy')
    if not np.isnan(trial.metrics.get_best_value('val_loss')) else np.nan, best))
    print(best_accuracy)
if __name__ == '__main__':
    load_results()