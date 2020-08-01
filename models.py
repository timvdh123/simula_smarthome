from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.python.ops import math_ops

class WeightedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    """Sparse Categorical loss, with weights added for imbalanced dataset. Meant to prevent the idle
    activity from always being predicted, as this activity appears more frequently.
    """
    def __call__(self, y_true, y_pred, sample_weight=None):
        sw = K.cast_to_floatx(K.equal(y_true, 0.0))*0.01
        sw = math_ops.add(sw, K.cast_to_floatx(K.not_equal(y_true, 0.0))*1)
        return super().__call__(y_true, y_pred, sample_weight=sw)

class WeightedBinaryCrossentropy(BinaryCrossentropy):
    """A balanced BinaryCrossentropy loss. Helps to learn models for sensors with very few
    activations.
    """
    def __call__(self, y_true, y_pred, sample_weight=None):
        """ Automatically calculates the weight of 0-1 activations using the number of
        activations in y_true.
        """
        num_ones = K.sum(y_true)
        num_zeros = K.sum(K.cast_to_floatx(K.not_equal(y_true, 0.0)))

        sw = K.cast_to_floatx(K.equal(y_true, 0.0))*(num_ones/num_zeros)
        sw = math_ops.add(sw, K.cast_to_floatx(K.not_equal(y_true, 0.0))*1)
        return super().__call__(y_true, y_pred, sample_weight=sw)


def single_sensor_multistep_future(
        timesteps,
        future_timesteps,
        n_features,
        input_activation='tanh',
        input_n_units=10,
        hidden_layers=1,
        hidden_layer_activation='relu',
        hidden_layer_units=5,
        output_activcation='sigmoid',
        learning_rate=0.00476,
        second_layer_input=None
):
    """Creates a model which can be used to learn the activations of a single sensor. A variable
    number of future timesteps can be predicted, although predicting 1 timestep ahead is more
    accurate.
    """
    model = models.Sequential()
    model.add(layers.GRU(input_n_units, activation=input_activation, input_shape=(timesteps,
                                                                                n_features),
                   return_sequences=True))
    if second_layer_input is None:
        second_layer_input = input_n_units//2
    model.add(layers.GRU(second_layer_input, activation=input_activation,
                   return_sequences=False))

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_layer_units, activation=hidden_layer_activation))
    model.add(layers.Dense(future_timesteps, activation=output_activcation))
    adam = optimizers.Adam(learning_rate)
    loss = WeightedBinaryCrossentropy()

    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def activity_synthesis_vector_output(
        timesteps,
        future_timesteps,
        n_features,
        n_activities,
        input_activation='tanh',
        input_n_units=10,
        hidden_layers=1,
        hidden_layer_activation='softmax',
        hidden_layer_units=5,
        output_activcation='softmax',
        learning_rate=0.00476,
        second_layer_input=None,
):
    """Creates a model which can be used to learn a pattern of activities in a day.

    The input should have shape [#Samples, #Past timesteps, #features]

    The output is a [#Samples, #Future Timesteps, #Activities] shape where the #Activities is a
    vector (which sums to 1) of probabilities of the next activity being activated.

    """
    model = models.Sequential()
    model.add(layers.GRU(input_n_units, activation=input_activation, input_shape=(timesteps, n_features),
                   return_sequences=False))
    model.add(layers.RepeatVector(future_timesteps))
    if second_layer_input is None:
        second_layer_input = input_n_units//2
    model.add(layers.GRU(second_layer_input, activation=input_activation,
                   return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(n_activities, activation='softmax')))
    adam = optimizers.Adam(learning_rate)
    loss = WeightedSparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def activity_synthesis_convolutional(
        timesteps,
        future_timesteps,
        n_features,
        n_activities,
        learning_rate=0.00476
):
    """A Convolutional Network which can be used instead of the recurrent models above to predict
    activities.
    """
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64,
                            kernel_size=2,
                            activation='relu',
                            input_shape=(timesteps, n_features)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(future_timesteps))

    model.add(layers.TimeDistributed(layers.Dense(n_activities, activation='softmax')))
    adam = optimizers.Adam(learning_rate)
    loss = WeightedSparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


def single_sensor_multistep_future_encoder_decoder(
        timesteps,
        future_timesteps,
        n_features,
        input_activation='tanh',
        input_n_units=10,
        encoder_extra_lstm=True,
        encoder_extra_lstm_units=10,
        encoder_extra_lstm_activation='tanh',
        decoder_lstm_units=10,
        decoder_lstm_activation='tanh',
        decoder_hidden_layers=1,
        decoder_hidden_layer_activation='tanh',
        decoder_hidden_layer_units=5,
        output_activcation='sigmoid',
        learning_rate=0.00476
):
    """An encoder-decoder model which can be used to predict future sensor values.
    """
    model = models.Sequential()
    # encoder
    input_return_sequences=encoder_extra_lstm
    model.add(layers.LSTM(input_n_units, activation=input_activation, input_shape=(timesteps, n_features),
                   return_sequences=input_return_sequences))
    if encoder_extra_lstm:
        model.add(layers.LSTM(encoder_extra_lstm_units, activation=encoder_extra_lstm_activation, return_sequences=False))
    model.add(layers.RepeatVector(future_timesteps))
    #decoder
    model.add(layers.LSTM(decoder_lstm_units, activation=decoder_lstm_activation, return_sequences=True))
    for _ in range(decoder_hidden_layers):
        model.add(layers.Dense(decoder_hidden_layer_units, activation=decoder_hidden_layer_activation))
    model.add(layers.TimeDistributed(layers.Dense(1, activation=output_activcation)))
    model.add(layers.Reshape((future_timesteps, )))
    adam = optimizers.Adam(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model
