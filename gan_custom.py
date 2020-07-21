import os

from fastdtw import fastdtw
from tensorflow.keras import layers, optimizers, models, losses, regularizers, Input, Model
from tqdm import tqdm

from discretize import Dataset

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def create_generator(
        timesteps,
        future_timesteps,
        n_features,
        input_activation='tanh',
        input_n_units=100,
        hidden_layers=1,
        hidden_layer_activation='relu',
        hidden_layer_units=20,
        output_activcation='sigmoid',
        learning_rate=0.00476
):
    model = models.Sequential()
    model.add(layers.GRU(input_n_units, activation=input_activation,
                         input_shape=(timesteps, n_features),
                         return_sequences=True))
    model.add(layers.GRU(input_n_units // 2, activation=input_activation,
                         return_sequences=False))

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_layer_units, activation=hidden_layer_activation))
    model.add(layers.Dense(future_timesteps, activation=output_activcation))
    adam = optimizers.Adam(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


# def create_discriminator(
#         future_timesteps,
#         n_features,
# ):
#     model = models.Sequential()
#     model.add(layers.GRU(future_timesteps, activation='relu', input_shape=(future_timesteps, n_features),
#                          return_sequences=False, return_state=False, unroll=True))
#     model.add(layers.Reshape((future_timesteps // 2, future_timesteps // 2)))
#     model.add(layers.Conv1D(16, 3, 2, "same"))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Conv1D(32, 3, 2, "same"))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Conv1D(64, 3, 2, "same"))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Conv1D(128, 3, 1, "same"))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
#     model.summary()
#     return model
def create_discriminator(
        future_timesteps,
        n_features,
):
    model = models.Sequential()
    model.add(layers.GRU(future_timesteps, activation='relu', input_shape=(future_timesteps, n_features),
                         return_sequences=True))
    model.add(layers.GRU(future_timesteps, activation='relu',
                         return_sequences=False))
    model.add(layers.Dense(future_timesteps, activation='relu'))
    model.add(layers.Dense(future_timesteps, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    return model


def generate_data(d: Dataset, dt=600, window_size=144, sensor_id=24):
    sensor_data = d.sensor_values_reshape(dt)[[sensor_id]]
    seconds_in_day = 24 * 60 * 60
    seconds_past_midnight = sensor_data.index.hour * 3600 + sensor_data.index.minute * 60 + \
                            sensor_data.index.second
    sensor_data['sin_time'] = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)
    sensor_data['cos_time'] = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)

    data = []
    for i in range(len(sensor_data) - window_size):
        c = sensor_data.shift(-1 * i)
        data.append(c.values[:window_size])
    return data


def prepare_data_future_steps(d, window_size=70, dt=60,
                              with_time=True, future_steps=20, sensor_id=6):
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
    return data[window_size:-future_steps, :, :], output[window_size:-future_steps, :, 0]


def create_gan(discriminator, generator, window_size, future_size):
    discriminator.trainable = False
    gan_input = Input(shape=(window_size, 3))
    x = generator(gan_input)
    # x = tf.round(x)
    x = layers.Reshape((future_size, 1))(x)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def test_train_val_split(X, y, dt, n_val_days, n_test_days):
    X_train = X[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24, :, :]
    X_val = X[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24, :,
            :]
    X_test = X[-n_test_days * (3600 // dt) * 24:, :, :]
    y_train = y[:-1 * (n_val_days + n_test_days) * (3600 // dt) * 24, :]
    y_val = y[-(n_val_days + n_test_days) * (3600 // dt) * 24:-n_test_days * (3600 // dt) * 24, :]
    y_test = y[-n_test_days * (3600 // dt) * 24:, :]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def generate_noise(batch_size, window_size):
    noise = np.random.random((batch_size, window_size, 3))
    noise[:, :, 0] = np.round(noise[:, :, 0])
    noise[:, :, 1] *= 24 * 60 * 60
    noise[:, :, 2] = np.cos(2 * np.pi * noise[:, :, 1] / 24 * 60 * 60)
    noise[:, :, 1] = np.sin(2 * np.pi * noise[:, :, 1] / 24 * 60 * 60)
    assert noise.shape == (batch_size, window_size, 3)
    return noise

def predict_validation_set(generator, X_val, y_val):
    y_val_predicted = generator.predict(X_val)
    y_val_predicted = (y_val_predicted > 0.5)
    for i in range(0, y_val.shape[0], 20):
        distance, path = fastdtw(y_val[i, :], y_val_predicted[i, :])
        fig, ax = plt.subplots()
        ax.plot(y_val[i, :], label='Actual')
        ax.plot(y_val_predicted[i, :], label='Predicted')
        plt.legend()
        plt.title("Accuracy = %3.2f\n DTW dist = %3.2f" % (
            tf.metrics.binary_accuracy(y_val[i, :], y_val_predicted[i, :]),
            distance
        ))
        plt.show()


def training(epochs=1, batch_size=32):
    future_timesteps = 144
    window_size = 360
    dt = 600

    # Loading the data
    bathroom1 = Dataset.parse('dataset/', 'bathroom1')
    kitchen1 = Dataset.parse('dataset/', 'kitchen1')
    combined1 = bathroom1.combine(kitchen1)

    X, y = prepare_data_future_steps(combined1, dt=dt, window_size=window_size,
                                     future_steps=future_timesteps, sensor_id=6)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = test_train_val_split(X, y, dt, 1, 2)

    # Creating GAN
    generator = create_generator(window_size, future_timesteps, X.shape[2])
    generator.load_weights('generator.h5')
    discriminator = create_discriminator(future_timesteps, 1)
    discriminator.load_weights('discriminator.h5')
    gan = create_gan(discriminator, generator, window_size, future_timesteps)
    gan.load_weights('gan.h5')
    for e in range(1, epochs + 1):
        generator.save('generator.h5')
        discriminator.save('discriminator.h5')
        gan.save('gan.h5')
        print("Epoch %d" % e)
        predict_validation_set(generator, X_val, y_val)
        for _ in tqdm(range(batch_size)):
            # generate  random noise as an input  to  initialize the  generator
            noise = generate_noise(batch_size, window_size)
            # Get a random set of  real images
            image_batch = y_train[np.random.randint(low=0, high=y_train.shape[0], size=batch_size)]

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            generated_images = tf.round(generated_images)

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1

            X = X.reshape(-1, future_timesteps, 1)

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)


            # Tricking the noised input of the Generator as real data
            noise = generate_noise(batch_size, window_size)
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)


if __name__ == '__main__':
    training(20, 32)
