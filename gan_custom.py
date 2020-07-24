import os

from fastdtw import fastdtw
from tensorflow.keras import layers, optimizers, models, losses, regularizers, Input, Model
from tqdm import tqdm

from discretize import Dataset

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lstm import save_model, evaluate_model, prepare_data_future_steps


def create_generator(
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
    model.add(layers.GRU(input_n_units//2, activation=input_activation,
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


def create_generator_seq2seq(
        timesteps,
        future_timesteps,
        n_features
):
    model = models.Sequential()
    model.add(layers.LSTM(150, input_shape=(timesteps, n_features)))
    model.add(layers.RepeatVector(1))
    model.add(layers.LSTM(150, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(future_timesteps, activation='sigmoid')))
    adam = optimizers.Adam(1e-4)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def create_discriminator(
        future_timesteps,
        n_features,
):
    model = models.Sequential()
    model.add(layers.GRU(future_timesteps, activation='relu', input_shape=(future_timesteps, n_features),
                         return_sequences=False))
    # model.add(layers.GRU(future_timesteps, activation='relu',
    #                      return_sequences=False))
    model.add(layers.Dense(future_timesteps, activation='relu'))
    model.add(layers.Dense(future_timesteps, activation='relu'))
    # model.add(layers.Dense(future_timesteps, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    return model

def create_gan(discriminator, generator, window_size, future_size):
    discriminator.trainable = False
    gan_input = Input(shape=(window_size, 3))
    x = generator(gan_input)
    # x = (x > 0.5)
    # x = tf.round(x)
    x = layers.Reshape((future_size, 1))(x)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    if len(y_val_predicted.shape) == 3:
        y_val_predicted = y_val_predicted.reshape(y_val.shape[0], y_val.shape[1])
    i = np.random.randint(0, y_val.shape[0])
    distance, path = fastdtw(y_val[i, :], y_val_predicted[i, :])
    fig, ax = plt.subplots()
    ax.plot(y_val[i, :], label='Actual')
    ax.plot(y_val_predicted[i, :] > 0.5, label='Predicted')
    plt.legend()
    plt.title("Accuracy = %3.2f\n DTW dist = %3.2f" % (
        tf.metrics.binary_accuracy(y_val[i, :], y_val_predicted[i, :]),
        distance
    ))
    plt.show()

def training(d, model_name,
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
    y = y[:, :, 0]
    y = y * 0.9 + 0.05

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = test_train_val_split(X, y, dt, 2, 2)

        # Creating GAN
    generator = create_generator_seq2seq(window_size, future_steps, X.shape[2])
    discriminator = create_discriminator(future_steps, 1)
    if load_weights:
        try:
            discriminator.load_weights('discriminator_%d.h5' % sensor_id)
        except Exception as e:
            print("Could not load weights")
        try:
            generator.load_weights('generator_%d.h5' % sensor_id)
        except Exception as e:
            print("Could not load weights")
    gan = create_gan(discriminator, generator, window_size, future_steps)
    if load_weights:
        try:
            gan.load_weights('gan_%d.h5' % sensor_id)
        except Exception as e:
            print("Could not load weights")

    for e in range(1, epochs + 1):
        if e % 50 == 0:
            generator.save('generator_sensor_%d_epoch_%d.h5' % (sensor_id, e))
            discriminator.save('discriminator_sensor_%d_epoch_%d.h5' % (sensor_id, e))
            gan.save('gan_sensor_%d_epoch_%d.h5' % (sensor_id, e))
            predict_validation_set(generator, X_val, y_val)
        print("Epoch %d" % e)
        disciminator_indices = np.arange(0, y_train.shape[0])
        np.random.shuffle(disciminator_indices)
        generator_indices = np.arange(0, y_train.shape[0])
        np.random.shuffle(generator_indices)

        for i in tqdm(range(0, len(y_train), batch)):

            # generate  random noise as an input  to  initialize the  generator
            # noise = generate_noise(batch, window_size)
            noise = X_train[disciminator_indices[i:i+batch]]

            # Get a random set of  real images
            image_batch = y_train[disciminator_indices[i:i+batch]]

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            if len(generated_images.shape) == 3:
                generated_images = generated_images.reshape(image_batch.shape[0],
                                                            image_batch.shape[1])
            # generated_images = tf.round(generated_images)

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * len(disciminator_indices[i:i+batch]))
            y_dis[:batch] = 1

            X = X.reshape(-1, future_steps, 1)

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            metrics = discriminator.train_on_batch(X, y_dis, return_dict=True)
            # print("Discriminator [Loss=%3.2f, Accuracy=%3.2f]" % (metrics['loss'],
            #                                                       metrics['accuracy']))

            # generator.train_on_batch(X_train[generator_indices[i:i+batch]],
            #                          y_train[generator_indices[i:i+batch]])

            # Tricking the noised input of the Generator as real data
            # noise = generate_noise(batch, window_size)
            noise = X_train[generator_indices[i:i+batch]]
            y_gen = np.ones(len(generator_indices[i:i+batch]))

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            metrics = gan.train_on_batch(noise, y_gen, return_dict=True)
            # print("GAN [Loss=%3.2f, Accuracy=%3.2f]" % (metrics['loss'],
            #                                             metrics['accuracy']))
    folder = save_model(model_name, generator, {}, kwargs)
    evaluate_model(generator, model_name, sensor_id, X_test, y_test, None,
                   save_folder=folder)