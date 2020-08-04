import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


def distance_datetime(X, Y):
    X_start = pd.to_datetime(X[0])
    Y_start = pd.to_datetime(Y[0])
    X_start = X_start.hour * 60 + X_start.minute
    Y_start = Y_start.hour * 60 + Y_start.minute
    diff = np.abs(X_start - Y_start)
    if diff > 12*60:
        diff = 24*60 - diff
    return diff / 12*60

def distance_duration_start(X, Y):
    X_start = pd.to_datetime(X[0])
    Y_start = pd.to_datetime(Y[0])
    X_start = X_start.hour * 60 + X_start.minute
    Y_start = Y_start.hour * 60 + Y_start.minute
    diff = np.abs(X_start - Y_start)
    if diff > 12*60:
        diff = 24*60 - diff
    return (((diff/12*60)**1 + np.abs(X[1]-Y[1])**1))

def LOF(d, features, n_neighbours=5):
    """Run LOF on each sensor separately

    :param d: the dataset
    :param features: some combination of ['start_time', 'duration']
    :param n_neighbours: the number of neighbours used in LOF
    """
    for id in d.sensor_data.id.unique():
        sensor_data = d.sensor_data.loc[d.sensor_data.id == id]
        sensors_start_time = sensor_data['start_time'].dt.hour * 60 + sensor_data['start_time'].dt.minute
        sensors_duration = (sensor_data['end_time'] - sensor_data['start_time']).astype('timedelta64[s]')
        if features == ['start_time']:
            clf = LocalOutlierFactor(n_neighbors=n_neighbours, metric=distance_datetime, n_jobs=8)
            clf.fit_predict(sensor_data['start_time'].astype(int).values.reshape(-1, 1))
        elif features == ['duration']:
            clf = LocalOutlierFactor(n_neighbors=n_neighbours, metric='euclidean', n_jobs=8)
            clf.fit_predict(MinMaxScaler().fit_transform(sensors_duration.values.reshape(-1, 1)))
        elif features == ['start_time', 'duration'] or features == ['duration', 'start_time']:
            clf = LocalOutlierFactor(n_neighbors=2, metric=distance_duration_start, n_jobs=8)
            clf.fit_predict(
                np.vstack((sensor_data['start_time'].astype(int).values, sensors_duration.values.reshape(-1, 1).T[0])).T)
        else:
            raise ValueError("Features %s not supported" % features)
        X_scores = clf.negative_outlier_factor_
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        print(radius)
        fig, ax = plt.subplots()
        ax.set_title('Sensor duration distribution - %s\n%s' % (features, d.lookup_sensor_id(id)))
        sns.scatterplot(sensors_start_time, sensors_duration, ax=ax, label='Sensor values')
        plt.scatter(sensors_start_time, sensors_duration, s=1000 * radius, edgecolors='r',
                    facecolors='none', label='Outlier scores')
        ax.legend()
        ax.set_ylabel('Sensor activation time [s]')
        plt.show()

def isolation_forest(d):
    """Use an isolation forest to find outliers for each sensor separately.

    :param d: the dataset
    :type d:
    :return: nothing
    :rtype:
    """
    for id in d.sensor_data.id.unique():
        sensor_data = d.sensor_data.loc[d.sensor_data.id == id]
        sensors_start_time = sensor_data['start_time'].dt.hour * 60 + sensor_data[
            'start_time'].dt.minute
        sensors_duration = (sensor_data['end_time'] - sensor_data['start_time']).astype(
            'timedelta64[s]')
        X = np.vstack((
            sensor_data['start_time'].astype(int).values,
            sensors_duration.values.reshape(-1, 1).T[0],
        )).T
        clf = IsolationForest(random_state=42)
        clf.fit(X)
        X_scores = clf.decision_function(X)
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        radius = np.array([r if r > 0.5 else 0 for r in radius])
        fig, ax = plt.subplots()
        ax.set_title('Sensor duration distribution - %s\n%s' % (['duration', 'start_time'],
                                                                d.lookup_sensor_id(
            id)))
        sns.scatterplot(sensors_start_time, sensors_duration, ax=ax, label='Sensor values')
        plt.scatter(sensors_start_time, sensors_duration, s=1000 * radius, edgecolors='r',
                    facecolors='none', label='Outlier scores')
        ax.legend()
        ax.set_ylabel('Sensor activation time [s]')
        plt.show()

def isolation_forest_all(d):
    """Use an isolation forest to find outliers with all data combined. Combine the results in a
    single plot

    :param d: the dataset
    :type d:
    :return: nothing
    :rtype:
    """
    sensor_data = d.sensor_data
    sensors_start_time = sensor_data['start_time'].dt.hour * 60 + sensor_data[
        'start_time'].dt.minute
    sensors_duration = (sensor_data['end_time'] - sensor_data['start_time']).astype(
        'timedelta64[s]')
    X = np.vstack((
        sensor_data['start_time'].astype(int).values,
        sensors_duration.values.reshape(-1, 1).T[0],
        sensor_data['id']
    )).T
    clf = IsolationForest(random_state=42)
    clf.fit(X)
    X_scores = clf.decision_function(X)
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    decision_function = np.array([1 if r > 0.5 else 0 for r in radius])
    sensor_data['anomaly'] = decision_function

    for id in d.sensor_data.id.unique():
        sensor_data = d.sensor_data.loc[d.sensor_data.id == id]
        sensors_start_time = sensor_data['start_time'].dt.hour * 60 + sensor_data[
            'start_time'].dt.minute
        sensors_duration = (sensor_data['end_time'] - sensor_data['start_time']).astype(
            'timedelta64[s]')
        fig, ax = plt.subplots()
        ax.set_title('Sensor duration distribution - %s\n%s' % (['duration', 'start_time'],
                                                                d.lookup_sensor_id(
            id)))
        sns.scatterplot(sensors_start_time, sensors_duration, ax=ax, label='Sensor values')
        plt.scatter(
            sensors_start_time,
            sensors_duration,
            s=1000 * sensor_data['anomaly'],
            edgecolors='r', facecolors='none', label='Outlier scores')
        ax.legend()
        ax.set_ylabel('Sensor activation time [s]')
        plt.show()

    radius = np.array([r if r > 0.5 else 0 for r in radius])


def LOF_all_data(d):
    """Use LOF to find outliers with all data combined. Combine the results in a
    single plot

    :param d: the dataset
    :type d:
    :return: nothing
    :rtype:
    """
    sensor_data = d.sensor_data
    sensors_start_time = sensor_data['start_time'].dt.hour * 60 + sensor_data[
        'start_time'].dt.minute
    sensors_duration = (sensor_data['end_time'] - sensor_data['start_time']).astype(
        'timedelta64[s]')
    clf = LocalOutlierFactor(n_neighbors=5, metric=distance_datetime, n_jobs=8)
    clf.fit_predict(sensor_data['start_time'].astype(int).values.reshape(-1, 1))

    X_scores = clf.negative_outlier_factor_
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    print(radius)
    fig, ax = plt.subplots()
    ax.set_title('Sensor duration distribution\nAll data')
    sns.scatterplot(sensors_start_time, sensors_duration, ax=ax, label='Sensor values')
    plt.scatter(sensors_start_time, sensors_duration, s=1000 * radius, edgecolors='r',
                facecolors='none', label='Outlier scores')
    ax.legend()
    ax.set_ylabel('Sensor activation time [s]')
    plt.show()

