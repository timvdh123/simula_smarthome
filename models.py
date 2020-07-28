import tensorflow as tf
from fastdtw import fastdtw
from tensorflow.keras import layers, optimizers, models, losses, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.ops import math_ops


def custom_loss(loss_metric, extra_weight_1=1000, extra_weight_all_zeros=100):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    @tf.function
    def loss(y_true, y_pred):
        rounded = K.round(y_pred)
        bce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        ones = K.greater(0.5, y_true)
        not_equal = K.not_equal(ones, K.greater(0.5, rounded))
        # Add extra loss for 1s which were predicted as a 0
        bce = tf.math.multiply(bce, (1 + K.cast_to_floatx(not_equal) * extra_weight_1))
        mean_per_batch = K.mean(bce, axis=1)
        all_zeros = K.equal(0.0, K.sum(rounded, axis=1))
        #Add extra weight to sequences of all 0s
        mean_per_batch = tf.math.multiply(mean_per_batch, (1 + K.cast_to_floatx(
            all_zeros)*extra_weight_all_zeros))
        return K.mean(mean_per_batch, axis=-1)

    # Return a function
    return loss

@tf.function
def sequence_matching_loss(y_true, y_pred):
    # #Two components: Number of activations and d
    dist = 0.0
    tf.print(K.shape(y_true))

    batch_size = K.shape(y_true)[0]
    prediction_length = K.shape(y_true)[1]
    slice_size = prediction_length//6

    def get_bc_slices(y_target, y_predicted):
        slices = []
        for slice_start in range(0, prediction_length, slice_size):
            y_predicted_slice = tf.slice(y_predicted, slice_start, slice_size)
            y_target_slice = tf.slice(y_target, slice_start, slice_size)
            slices.append(K.binary_crossentropy(y_target_slice, y_predicted_slice))

    rolls = []
    for roll_size in range(-2, 2):
        y_pred_rolled = tf.roll(y_pred, roll_size, axis=1)
        for i in range(batch_size):
            slice_bc = get_bc_slices(y_true[i], y_pred_rolled[i])
            rolls.append(slice_bc)
        rolls = tf.convert_to_tensor(rolls)
        tf.print(rolls)
        s1 = K.sum(y_true[i])
        s2 = K.sum(K.cast_to_floatx(K.greater(0.5, y_pred[i])))
        # tf.print(s1)
        # tf.print(s2)
        dist += K.abs((s1 - s2))

        w1 = tf.where(K.greater(0.5, y_true[i]))
        w2 = tf.where(K.greater(0.5, y_pred[i]))
        # tf.print(w1)
        # tf.print(w2)

        distance = tf.reduce_sum(tf.abs(tf.subtract(w1, tf.expand_dims(w2, 1))),
                                 axis=2)
        # tf.print(distance)
        _, top_k_indices = tf.compat.v1.nn.top_k(tf.negative(distance), k=1)
        # tf.print(top_k_indices)
        # _dist, _path = fastdtw(y_true[i].eval(), y_pred[i].eval())
        # dist += _dist
    return dist

class WeightedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __call__(self, y_true, y_pred, sample_weight=None):
        sw = K.cast_to_floatx(K.equal(y_true, 0.0))*0.01
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
    bc = losses.BinaryCrossentropy()
    # loss = sequence_matching_loss(bc)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
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

def get_model(timesteps, n_features, lr):
    lstm_autoencoder = models.Sequential()
    lstm_autoencoder.add(layers.LSTM(10, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_autoencoder.add(layers.LSTM(6, activation='relu', return_sequences=True))
    lstm_autoencoder.add(layers.LSTM(1, activation='relu'))
    lstm_autoencoder.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_autoencoder.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_autoencoder.add(layers.Dense(n_features, activation='sigmoid'))
    adam = optimizers.Adam(lr)
    lstm_autoencoder.compile(loss='binary_crossentropy', optimizer=adam)
    lstm_autoencoder.summary()
    return lstm_autoencoder

def get_model_individual_sensor(timesteps, n_features, lr):
    lstm_regular = models.Sequential()
    lstm_regular.add(layers.LSTM(15, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_regular.add(layers.LSTM(8, activation='relu', return_sequences=False))
    lstm_regular.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(layers.Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr)
    lstm_regular.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    lstm_regular.summary()
    return lstm_regular

def get_model_all_sensors(timesteps, n_features, lr):
    lstm_regular = models.Sequential()
    lstm_regular.add(layers.LSTM(15, activation='relu', input_shape=(timesteps, n_features),
                              return_sequences=True))
    lstm_regular.add(layers.Dropout(0.3))
    lstm_regular.add(layers.LSTM(8, activation='relu', return_sequences=False))
    lstm_regular.add(layers.Dropout(0.3))
    lstm_regular.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    lstm_regular.add(layers.Dropout(0.3))
    lstm_regular.add(layers.Dense(10, kernel_initializer='glorot_normal', activation='relu',
                           kernel_regularizer=l1(1e-4)))
    lstm_regular.add(layers.Dropout(0.3))
    lstm_regular.add(layers.Dense(n_features, activation='sigmoid'))
    adam = optimizers.Adam(lr)
    lstm_regular.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    lstm_regular.summary()
    return lstm_regular
