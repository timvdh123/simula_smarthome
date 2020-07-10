# %%
import csv
from collections import Counter

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from hmmlearn import hmm
import hmmlearn.utils
import hmmlearn.stats

from similarity import LOF, isolation_forest, isolation_forest_all


class Data:
    def __init__(self, prefix, sensor_data, activity_data, sensor_labels, activity_labels):
        self.prefix: pd.DataFrame = prefix
        self.sensor_data: pd.DataFrame = sensor_data
        self.activity_data: pd.DataFrame = activity_data
        self.sensor_labels: pd.DataFrame = sensor_labels
        self.activity_labels: pd.DataFrame = activity_labels

    def combine(self, other):
        sensor_data = pd.concat((self.sensor_data, other.sensor_data))
        activity_data = pd.concat((self.activity_data, other.activity_data))

        return Data(
            'combined[%s/%s]' % (self.prefix, other.prefix),
            sensor_data,
            activity_data,
            self.sensor_labels.copy(),
            self.activity_labels.copy()
        )

    def __set_previous_activation_times(self):
        for id in self.sensor_data.id.unique():
            sensor_data = self.sensor_data.loc[self.sensor_data.id == id]
            sorted_start_time = sensor_data.sort_values(by=['start_time'])
            sorted_start_time['prev_end_time'] = sorted_start_time.shift(1)['end_time']


    def lookup_sensor_id(self, id):
        values = self.sensor_labels.loc[self.sensor_labels.id == id, 'label']
        if len(values) == 0:
            raise ValueError("No such sensor with id=%s" % id)
        return values.values[0]

    def lookup_activity_id(self, id):
        values = self.activity_labels.loc[self.activity_labels.id == id, 'label']
        if len(values) == 0:
            raise ValueError("No such activity with id=%s" % id)
        return values.values[0]

    @staticmethod
    def parse(prefix):
        df = pd.read_csv('{:s}.ss.csv'.format(prefix))
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        df_as = pd.read_csv('{:s}.as.csv'.format(prefix))
        df_as['start_time'] = pd.to_datetime(df_as['start_time'])
        df_as['end_time'] = pd.to_datetime(df_as['end_time'])

        activity_labels = pd.read_csv('{:s}.activity_labels.csv'.format(prefix), delimiter=',,')
        sensor_labels = pd.read_csv('{:s}.sensor_labels.csv'.format(prefix), delimiter=',,')
        return Data(prefix, df, df_as, sensor_labels, activity_labels)

    def plot_sensor_values(self):
        for id in self.sensor_data.id.unique():
            print(id)
            data_weekday = self.sensor_data.loc[(self.sensor_data.id == id) & (
                    self.sensor_data.start_time.dt.dayofweek < 5)]
            data_weekend = self.sensor_data.loc[(self.sensor_data.id == id) & (self.sensor_data.start_time.dt.dayofweek >= 5)]
            sensors_start_time_weekday = data_weekday['start_time'].dt.hour*60 + data_weekday['start_time'].dt.minute
            sensors_start_time_weekend = data_weekend['start_time'].dt.hour*60 + data_weekend[
                'start_time'].dt.minute
            sensors_duration_weekday = (data_weekday['end_time'] - data_weekday['start_time']).astype('timedelta64[s]')
            sensors_duration_weekend = (data_weekend['end_time'] - data_weekend['start_time']).astype('timedelta64[s]')
            fig, ax = plt.subplots()
            ax.set_title('Sensor activation distribution\n%s' % self.lookup_sensor_id(id))
            sns.distplot(sensors_start_time_weekday, ax=ax, label='weekday')
            sns.distplot(sensors_start_time_weekend, ax=ax, label='weekend')
            ax.legend()
            plt.show()
            fig, ax = plt.subplots()
            ax.set_title('Sensor duration distribution\n%s' % self.lookup_sensor_id(id))
            sns.scatterplot(sensors_start_time_weekday, sensors_duration_weekday, ax=ax,
                            label='weekday')
            sns.scatterplot(sensors_start_time_weekend, sensors_duration_weekend, ax=ax,
                            label='weekend')
            ax.set_ylabel('Sensor activation time [s]')
            plt.show()


    def last_fired(self, dt = 60):
        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(60, 's')
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
        return table

    def sensor_values_reshape(self, dt=60):
        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(dt, 's')
        ).astype('datetime64[ns]')

        sensor_label_indices = list(self.sensor_data.id.unique())
        table = np.zeros((len(time), len(sensor_label_indices)))

        sorted_start_time = self.sensor_data.sort_values(by=['start_time'])
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

    def activity_reshape(self):
        time = np.arange(
            start=pd.Timestamp(self.sensor_data.start_time.min().date()),
            stop=pd.Timestamp(self.sensor_data.end_time.max().date() + pd.Timedelta(1, 'day')),
            step=pd.to_timedelta(60, 's')
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

        return table.astype(int)

    def learn_hmm(self):
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
        for id in self.sensor_data.id.unique():
            data = self.sensor_data.loc[self.sensor_data.id == id]
            print("\nSensor %s was activated %d times" % (self.lookup_sensor_id(id), len(data)))
            duration = (data['end_time'] - data['start_time']).astype('timedelta64[s]').astype(int)
            print("Mean activation time: %5.2f +- %3.2e s" % (duration.mean(), duration.std()))
# %%
if __name__ == '__main__':
    bathroom1 = Data.parse('bathroom1')
    kitchen1 = Data.parse('kitchen1')
    combined = bathroom1.combine(kitchen1)
    # combined.sensor_data_summary()
    # LOF_all_data(combined)
    # LOF(combined, ['duration'], 2)
    isolation_forest_all(combined)
    # combined.plot_sensor_values()

    bathroom2 = Data.parse('bathroom2')
    kitchen2 = Data.parse('kitchen2')
    combined2 = bathroom2.combine(kitchen2)
    # combined2.sensor_data_summary()
    # combined2.plot_sensor_values()
    # LOF(combined2)
    # d = combined.sensor_values_reshape()
    # d[8].head(4*24*60).plot()
    # plt.show()

    # model = combined.learn_hmm()
    # table = bathroom1.last_fired()
    # bathroom1.plot_sensor_values()
    # kitchen1.plot_sensor_values()
    # bathroom2 = Data.parse('bathroom2')
    # kitchen2 = Data.parse('kitchen2')
    # kitchen2.plot_sensor_values()


