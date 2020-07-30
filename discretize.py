# %%
import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import classification_report


class Dataset:
    """Class to parse and store the dataset.
    """
    def __init__(self, prefix, sensor_data, activity_data, sensor_labels, activity_labels):
        self.prefix: pd.DataFrame = prefix
        self.sensor_data: pd.DataFrame = sensor_data
        self.activity_data: pd.DataFrame = activity_data
        self.sensor_labels: pd.DataFrame = sensor_labels
        self.activity_labels: pd.DataFrame = activity_labels

    def combine(self, other):
        """Combines multiple datasets (different rooms) into a single dataset.
        """
        sensor_data = pd.concat((self.sensor_data, other.sensor_data))
        activity_data = pd.concat((self.activity_data, other.activity_data))

        return Dataset(
            'combined[%s/%s]' % (self.prefix, other.prefix),
            sensor_data,
            activity_data,
            self.sensor_labels.copy(),
            self.activity_labels.copy()
        )

    def _set_previous_activation_times(self):
        for id in self.sensor_data.id.unique():
            sensor_data = self.sensor_data.loc[self.sensor_data.id == id]
            sorted_start_time = sensor_data.sort_values(by=['start_time'])
            sorted_start_time['prev_end_time'] = sorted_start_time.shift(1)['end_time']


    def lookup_sensor_id(self, id: int) -> str:
        """Looks up the name of a sensor using the id.

        :param id: the sensor id (int)
        :type id: int
        :return: the name of the sensor, or a ValueError if there is no sensor with that ID.
        :rtype: str
        """
        values = self.sensor_labels.loc[self.sensor_labels.id == id, 'label']
        if len(values) == 0:
            raise ValueError("No such sensor with id=%s" % id)
        return values.values[0]

    def lookup_activity_id(self, id):
        """Looks up the name of an activity using the id.

        :param id: the activity id (int)
        :type id: int
        :return: the name of the activity, or a ValueError if there is no activity with that ID.
        :rtype: str
        """
        values = self.activity_labels.loc[self.activity_labels.id == id, 'label']
        if len(values) == 0:
            raise ValueError("No such activity with id=%s" % id)
        return values.values[0]

    @staticmethod
    def parse(path, prefix):
        """Parses a data file.

        :param path: the path to the folder that contains the file
        :type path: str
        :param prefix: the name of the room
        :type prefix: str
        :return: a Dataset containing the sensor data, activity labels, etc.
        """
        df = pd.read_csv('{:s}{:s}.ss.csv'.format(path, prefix))
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        df_as = pd.read_csv('{:s}{:s}.as.csv'.format(path, prefix))
        df_as['start_time'] = pd.to_datetime(df_as['start_time'])
        df_as['end_time'] = pd.to_datetime(df_as['end_time'])

        activity_labels = pd.read_csv('{:s}{:s}.activity_labels.csv'.format(path, prefix),
                                      delimiter=',,')
        sensor_labels = pd.read_csv('{:s}{:s}.sensor_labels.csv'.format(path, prefix),
                                    delimiter=',,')
        return Dataset(prefix, df, df_as, sensor_labels, activity_labels)

    def plot_sensor_values(self):
        for id in self.sensor_data.id.unique():
            data_weekday = self.sensor_data.loc[(self.sensor_data.id == id) & (
                    self.sensor_data.start_time.dt.dayofweek < 5)]
            data_weekend = self.sensor_data.loc[(self.sensor_data.id == id) & (self.sensor_data.start_time.dt.dayofweek >= 5)]
            sensors_start_time_weekday = data_weekday['start_time'].dt.hour*60 + data_weekday['start_time'].dt.minute
            sensors_start_time_weekend = data_weekend['start_time'].dt.hour*60 + data_weekend[
                'start_time'].dt.minute
            sensors_duration_weekday = (data_weekday['end_time'] - data_weekday['start_time']).astype('timedelta64[s]')
            sensors_duration_weekend = (data_weekend['end_time'] - data_weekend['start_time']).astype('timedelta64[s]')

            # Plot sensor activation times.
            fig, ax = plt.subplots()
            ax.set_title('Sensor activation distribution\n%s' % self.lookup_sensor_id(id))
            sns.distplot(sensors_start_time_weekday, ax=ax, label='weekday')
            sns.distplot(sensors_start_time_weekend, ax=ax, label='weekend')
            ax.legend()
            plt.show()

            # Plot sensor activation times and duration in a scatter plot.
            fig, ax = plt.subplots()
            ax.set_title('Sensor duration distribution\n%s' % self.lookup_sensor_id(id))
            sns.scatterplot(sensors_start_time_weekday, sensors_duration_weekday, ax=ax,
                            label='weekday')
            sns.scatterplot(sensors_start_time_weekend, sensors_duration_weekend, ax=ax,
                            label='weekend')
            ax.set_ylabel('Sensor activation time [s]')
            plt.show()


    def last_fired(self, dt = 60):
        """Convert the sensor data into the 'last fired' representation. Windows the data into
        dt second bins.
        """
        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(dt, 's')
        ).astype('datetime64[ns]')
        sensor_label_indices = list(self.sensor_data.id.unique())
        table = np.zeros((len(time), len(sensor_label_indices)))

        sorted_start_time = self.sensor_data.sort_values(by=['start_time'])
        sorted_start_time = sorted_start_time.reset_index()
        sorted_start_time['bin_start'] = np.digitize(sorted_start_time['start_time'].astype(int),
                                                     time.astype(int))
        sorted_start_time['bin_end'] = np.digitize(sorted_start_time['end_time'].astype(int),
                                                   time.astype(
            int))
        for _, row in sorted_start_time.iterrows():
            table[
                row.bin_start:,
                sensor_label_indices.index(row['id'])
            ] = 1
            table[
                row['bin_start']:,
                np.array([i for i in range(len(sensor_label_indices))
                          if i != sensor_label_indices.index(row['id'])])
            ] = 0
        return pd.DataFrame(table, columns=sensor_label_indices,
                            index=time)

    def sensor_values_reshape(self, dt=60):
        """Discretizes the sensor data into bins. Sensor value is 1 if a bin overlaps with
        start_time, end_time.
        """
        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(dt, 's')
        ).astype('datetime64[ns]')

        sensor_label_indices = list(self.sensor_data.id.unique())
        table = np.zeros((len(time), len(sensor_label_indices)))

        sorted_start_time = self.sensor_data.sort_values(by=[
            'start_time'])
        sorted_start_time['bin_start'] = np.digitize(sorted_start_time['start_time'].astype(int),
                                                     time.astype(int))
        sorted_start_time['bin_end'] = np.digitize(sorted_start_time['end_time'].astype(int),
                                                   time.astype(
            int))
        for _, row in sorted_start_time.iterrows():
            table[
                np.arange(row.bin_start, row.bin_end+1),
                sensor_label_indices.index(row['id'])
            ] = 1
        return pd.DataFrame(table, columns=sensor_label_indices,
                            index=time)

    def activity_reshape(self, dt=60):
        """Discretizes the activity data into bins. Value is activity_id if a bin overlaps with
        start_time, end_time."""

        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(dt, 's')
        ).astype('datetime64[ns]')

        table = np.zeros(len(time))

        sorted_start_time = self.activity_data.sort_values(by=['start_time'])
        sorted_start_time['bin_start'] = np.digitize(sorted_start_time['start_time'].astype(int),
                                                     time.astype(int))
        sorted_start_time['bin_end'] = np.digitize(sorted_start_time['end_time'].astype(int),
                                                   time.astype(
            int))
        for _, row in sorted_start_time.iterrows():
            table[
                np.arange(row.bin_start, row.bin_end),
            ] = row.id
        return pd.DataFrame(table.astype(int), columns=['activity'],
                     index=time)

    def learn_hmm(self):
        """Learns a multinomial hidden markovel model over the sensor data."""
        remodel = hmm.MultinomialHMM(n_components=self.sensor_data.id.nunique(),
                                     n_iter=100, tol=1e-8)
        table = self.last_fired()
        X = table.argmax(axis=1)
        X = X.reshape(-1, 1)
        X_test = X[0:60*24]
        X_train = X[60*24:]
        remodel.fit(X_train)
        Z = remodel.predict(X_test)
        y_true = X_test.reshape(1, -1)[0]
        print(Counter(Z))
        print(Counter(y_true))
        print(classification_report(y_true, Z))
        print(remodel.predict_proba(X_test))
        return remodel

    def sensor_data_summary(self):
        """Prints a summary of the data"""
        counts = {}
        for id in self.sensor_data.id.unique():
            counts[id] = len(self.sensor_data.loc[self.sensor_data.id == id])
        sorted_counts = list(counts.items())
        sorted_counts.sort(key = lambda count: count[1], reverse=True)

        for id, count in sorted_counts:
            data = self.sensor_data.loc[self.sensor_data.id == id]
            print("\nSensor[%d] %s was activated %d times" % (id, self.lookup_sensor_id(id), count))
            duration = (data['end_time'] - data['start_time']).astype('timedelta64[s]').astype(int)
            print("Mean activation time: %5.2f +- %3.2e s" % (duration.mean(), duration.std()))

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

