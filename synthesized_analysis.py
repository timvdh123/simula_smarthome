from typing import Tuple

import numpy as np
import pandas as pd
from entropy import sample_entropy
from scipy.stats import entropy

import matplotlib.pyplot as plt

def load_data(input: str, synthesized: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Loads a synthesized dataset.
    :param input: path to the input csv
    :param synthesized:
    :return: the parsed datasets
    """
    in_dataset = pd.read_csv(input, index_col=0)
    synthesized_dataset = pd.read_csv(synthesized, index_col=0)
    return in_dataset, synthesized_dataset

def ApEn(U, m, r) -> float:
    """Approximate_entropy calculation. Code from
    https://en.wikipedia.org/wiki/Approximate_entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def calc_distribution(sequence: np.array) -> np.array:
    """Calculates the distribution of sensor activations
    (1-#activations, #activations)/length(dataset)

    :param sequence:
    :type sequence:
    :return:
    :rtype:
    """
    p_0 = (sequence <= 0.5).sum()/len(sequence)
    p_1 = (sequence > 0.5).sum()/len(sequence)
    return np.array([p_0, p_1])

def get_entropy_information(input, synthesized):
    real, synthesized = load_data(input, synthesized)
    day = 24*4
    rows = []
    for sensor in synthesized.columns:
        sample_entropy_order=16
        real_entropy_dict = {'Sensor id': '%s (real)' % sensor}
        synthesized_entropy_dict = {'Sensor id': '%s (synthesized)' % sensor}

        sensor_data = synthesized[[sensor]].values.reshape(1, -1)[0]
        real_sensor_data = real[[sensor]].values.reshape(1, -1)[0]

        fig, ax = plt.subplots()

        pd.plotting.autocorrelation_plot(real_sensor_data, ax=ax, label='Real')
        pd.plotting.autocorrelation_plot(sensor_data, ax=ax, label='Synthesized')
        ax.set_xlim(0, len(sensor_data))
        ax.legend()
        ax.set_title("Sensor %s autocorrelation plot" % sensor)
        plt.show()



        real_entropy_dict['Sample entropy'] = sample_entropy(real_sensor_data, order=sample_entropy_order)
        synthesized_entropy_dict['Sample entropy'] = sample_entropy(sensor_data, order=sample_entropy_order)

        sample_entropy_real = [sample_entropy(real_sensor_data[i:(i+day)], order=sample_entropy_order)
                        for i in range(0, len(real_sensor_data), day)]
        sample_entropy_synthesized = [sample_entropy(sensor_data[i:(i+day)], order=sample_entropy_order)
                        for i in range(0, len(sensor_data), day)]
        sample_entropy_real = np.array(sample_entropy_real)
        sample_entropy_synthesized = np.array(sample_entropy_synthesized)
        sample_entropy_real = sample_entropy_real[~np.isnan(sample_entropy_real)]
        sample_entropy_synthesized = sample_entropy_synthesized[~np.isnan(sample_entropy_synthesized)]

        real_entropy_dict['Sample entropy 24 hours'] = np.mean(sample_entropy_real)
        synthesized_entropy_dict['Sample entropy 24 hours'] = np.mean(sample_entropy_synthesized)

        entropy_real = [entropy(pk=calc_distribution(real_sensor_data[i:(i+day)]))
                        for i in range(0, len(real_sensor_data), day)]
        entropy_synthesized = [entropy(pk=calc_distribution(sensor_data[i:(i+day)]))
                        for i in range(0, len(sensor_data), day)]
        entropy_real = np.array(entropy_real)
        entropy_synthesized = np.array(entropy_synthesized)

        entropy_real = entropy_real[~np.isnan(entropy_real)]
        entropy_synthesized = entropy_synthesized[~np.isnan(entropy_synthesized)]

        real_entropy_dict['Entropy 24 hours'] = np.mean(entropy_real)
        synthesized_entropy_dict['Entropy 24 hours'] = np.mean(entropy_synthesized)

        real_entropy_dict['Entropy'] = np.mean(entropy(pk=calc_distribution(real_sensor_data), base=2))
        synthesized_entropy_dict['Entropy'] = entropy(pk=calc_distribution(sensor_data), base=2)
        rows.append(real_entropy_dict)
        rows.append(synthesized_entropy_dict)

    df = pd.DataFrame(rows)
    df.to_csv('entropy.csv')
        # print("Approximate entropy {}".format(ApEn(sensor_data, 24*4, 3)))
