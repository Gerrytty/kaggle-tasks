import pandas as pd
from numpy import exp, array, random, dot
import numpy


def count(input, test_y, synaptic_weights):
    count = 0
    for iteration in range(1000):
        y = 1 / (1 + exp(-(dot(input[iteration], synaptic_weights))))
        if y >= 1:
            py = 1
        else:
            py = 0

        if py == test_y[iteration]:
            count += 1

    return count


if __name__ == "__main__":
    numpy.seterr('ignore')
    all_data = pd.read_csv("all_out.csv", nrows=2000)

    print(all_data.head())

    all_data.loc[all_data['kmeans'] != 4, 'kmeans'] = 0
    all_data.loc[all_data['kmeans'] == 4, 'kmeans'] = 1

    all_y = all_data['kmeans']

    all_data = all_data[['year', 'money', 'experience']]

    # all_data.loc[all_data['money'] < 10000, 'money'] = 0
    # all_data.loc[all_data['money'] >= 10000, 'money'] = 1
    #
    # all_data.loc[all_data['year'] < 18, 'year'] = 0
    # all_data.loc[all_data['year'] >= 18, 'year'] = 1

    print(all_data.head())

    train_input = all_data.to_numpy()[:1000]

    train_y = list(all_y.to_numpy()[:1000])
    train_y = array([train_y]).T

    test_input = all_data.to_numpy()[1000:]
    test_y = all_y.to_numpy()[1000:]

    synaptic_weights = (2 * random.random((3, 1)) - 1) / 2

    print(f"Weights before {synaptic_weights}")

    for iteration in range(100):
        output = 1 / (1 + exp(-(dot(train_input, synaptic_weights))))
        synaptic_weights += dot(train_input.T, (train_y - output) * output * (1 - output))

    print(f"Weights after {synaptic_weights}")

    print(f"rights answers on train data = {count(train_input, train_y, synaptic_weights) / 1000 * 100}%")
    print(f"rights answers on test = {count(test_input, test_y, synaptic_weights) / 1000 * 100}%")