from typing import Tuple

import numpy as np
import pandas as pd
from entropy import sample_entropy
from scipy.stats import entropy

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
    """Approximate_entropy calculation. From wikipedia"""

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

def print_entropy_information():
    real, synthesized = load_data('input.csv', 'output.csv')
    day = 24*4
    for sensor in synthesized.columns:
        print("-----------------Sensor {}-------------".format(sensor))
        sensor_data = synthesized[[sensor]].values.reshape(1, -1)[0]
        real_sensor_data = real[[sensor]].values.reshape(1, -1)[0]

        print("Sample entropy [Synthesized] {:.3f}".format(sample_entropy(sensor_data, order=8)))
        print("Sample entropy [Real] {:.3f}".format(sample_entropy(real_sensor_data, order=8)))

        sample_entropy_real = [sample_entropy(real_sensor_data[i:(i+day)], order=8)
                        for i in range(0, len(real_sensor_data), day)]
        sample_entropy_synthesized = [sample_entropy(sensor_data[i:(i+day)], order=8)
                        for i in range(0, len(sensor_data), day)]
        sample_entropy_real = np.array(sample_entropy_real)
        sample_entropy_synthesized = np.array(sample_entropy_synthesized)
        sample_entropy_real = sample_entropy_real[~np.isnan(sample_entropy_real)]
        sample_entropy_synthesized = sample_entropy_synthesized[~np.isnan(sample_entropy_synthesized)]

        print("Sample entropy 24 hour windows [Synthesized] {:.3f}".format(np.mean(sample_entropy_synthesized)))
        print("Sample entropy 24 hour windows [Real] {:.3f}".format(np.mean(sample_entropy_real)))


        entropy_real = [entropy(pk=calc_distribution(real_sensor_data[i:(i+day)]))
                        for i in range(0, len(real_sensor_data), day)]
        entropy_synthesized = [entropy(pk=calc_distribution(sensor_data[i:(i+day)]))
                        for i in range(0, len(sensor_data), day)]
        entropy_real = np.array(entropy_real)
        entropy_synthesized = np.array(entropy_synthesized)

        entropy_real = entropy_real[~np.isnan(entropy_real)]
        entropy_synthesized = entropy_synthesized[~np.isnan(entropy_synthesized)]

        print("Synthesized Entropy {:.3f}".format(entropy(pk=calc_distribution(sensor_data),
                                                         base=2)))
        print("Real Entropy {:.3f}".format(entropy(pk=calc_distribution(real_sensor_data), base=2)))

        print("Mean 24-hour Synthesized Entropy {:.3f}".format(np.mean(entropy_synthesized)))
        print("Mean 24-hour Real Entropy {:.3f}".format(np.mean(entropy_real)))

        entropy_synthesized = [entropy(
            pk=calc_distribution(sensor_data[i:(i+day)]),
            qk=calc_distribution(real_sensor_data)
        )
                        for i in range(0, len(sensor_data), day)]
        entropy_real = np.array(entropy_real)
        entropy_synthesized = np.array(entropy_synthesized)

        print("\n")
        # print("Approximate entropy {}".format(ApEn(sensor_data, 24*4, 3)))

if __name__ == '__main__':
    print_entropy_information()