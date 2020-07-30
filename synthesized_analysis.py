import numpy as np
import pandas as pd
from scipy.stats import entropy

def load_data(input, synthesized):
    in_dataset = pd.read_csv(input, index_col=0)
    synthesized_dataset = pd.read_csv(synthesized, index_col=0)
    return in_dataset, synthesized_dataset

def ApEn(U, m, r) -> float:
    """Approximate_entropy."""

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

def calc_distribution(sequence):
    p_0 = (sequence <= 0.5).sum()/len(sequence)
    p_1 = (sequence > 0.5).sum()/len(sequence)
    return np.array([p_0, p_1])

if __name__ == '__main__':
    real, synthesized = load_data('input.csv', 'output.csv')
    day = 24*4
    for sensor in synthesized.columns:
        print("-----------------Sensor {}-------------".format(sensor))
        sensor_data = synthesized[[sensor]].values.reshape(1, -1)[0]
        real_sensor_data = real[[sensor]].values.reshape(1, -1)[0]

        entropy_real = [entropy(pk=calc_distribution(real_sensor_data[i:(i+day)]))
                        for i in range(0, len(real_sensor_data), day)]
        entropy_synthesized = [entropy(pk=calc_distribution(sensor_data[i:(i+day)]))
                        for i in range(0, len(sensor_data), day)]
        entropy_real = np.array(entropy_real)
        entropy_synthesized = np.array(entropy_synthesized)

        entropy_real = entropy_real[~np.isnan(entropy_real)]
        entropy_synthesized = entropy_synthesized[~np.isnan(entropy_synthesized)]

        print("Synthesized Entropy {}".format(entropy(pk=calc_distribution(sensor_data), base=2)))
        print("Real Entropy {}".format(entropy(pk=calc_distribution(real_sensor_data), base=2)))

        print("Mean 24-hour Synthesized Entropy {}".format(np.mean(entropy_synthesized)))
        print("Mean 24-hour Real Entropy {}".format(np.mean(entropy_real)))

        entropy_synthesized = [entropy(
            pk=calc_distribution(sensor_data[i:(i+day)]),
            qk=calc_distribution(real_sensor_data)
        )
                        for i in range(0, len(sensor_data), day)]
        entropy_real = np.array(entropy_real)
        entropy_synthesized = np.array(entropy_synthesized)

        print("\n")
        # print("Approximate entropy {}".format(ApEn(sensor_data, 24*4, 3)))