from __future__ import division
import numpy as np
import Constants as cts
from scipy.spatial.distance import euclidean
import math

sigma_0 = 0.25
learning_rate_0 = 0.5
num_epochs = 2000


# Time varying learning rate
def get_learning_rate(time):
    return learning_rate_0 * math.exp((-1 * time) / 100)


# Time varying sigma value
def get_sigma_val(time):
    return sigma_0 * math.exp((-1 * time)/1000)


# Neighborhood function of the SOFM
def get_neighborhood_value(neuron_idx, winning_neuron_idx, time):
    neighborhood = math.exp((-1 * euclidean(neuron_idx, winning_neuron_idx) ** 2)/get_sigma_val(time))
    return neighborhood


# Function to compute the change in weight at every point in time
def compute_weight_change(neuron_idx, winning_neuron_idx, weight_vector, input_vector, time):
    dist_vector = input_vector - weight_vector
    learning_rate = get_learning_rate(time)
    neighborhood = get_neighborhood_value(neuron_idx, winning_neuron_idx, time)
    weight_updates_vector = learning_rate * neighborhood * dist_vector
    return weight_updates_vector


# Function to train the SOFM
def train_sofm(weights, symbol_code, attribute_code):
    print 'Training SOFM for classification'
    for epoch in range(num_epochs):
        print 'Epoch - ', epoch+1
        for data_idx in range(len(cts.ANIMALS)):
            input_vector = np.concatenate((symbol_code[data_idx], attribute_code[data_idx]))
            winning_neuron_idx = (0, 0)
            min_dist = euclidean(input_vector, weights[0, 0])
            # Identify the winning neuron
            for neuron_idx1 in range(cts.MAP_SIZE):
                for neuron_idx2 in range(cts.MAP_SIZE):
                    if (neuron_idx1, neuron_idx2) == (0, 0):
                        continue
                    dist = euclidean(weights[neuron_idx1, neuron_idx2], input_vector)
                    if dist < min_dist:
                        min_dist = dist
                        winning_neuron_idx = (neuron_idx1, neuron_idx2)
            # Update weights
            for neuron_idx1 in range(cts.MAP_SIZE):
                for neuron_idx2 in range(cts.MAP_SIZE):
                    weight_vector = weights[neuron_idx1, neuron_idx2]
                    weights[neuron_idx1, neuron_idx2] = weight_vector + compute_weight_change(
                        (neuron_idx1, neuron_idx2), winning_neuron_idx, weight_vector, input_vector, epoch+1)
    return weights


def test_sofm(weights, symbol_code, attribute_code, flag):
    print 'Generating output of the network for %s data...' % flag
    test_outcome = {}
    if flag == 'test':
        output_dict = cts.ANIMALS
    else:
        output_dict = cts.NEW_ANIMALS

    for data_idx in range(len(output_dict)):
        input_vector = np.concatenate((symbol_code[data_idx], attribute_code[data_idx]))
        winning_neuron_idx = (0, 0)
        max_output = np.dot(weights[0, 0].transpose(), input_vector)
        # Identify the winning neuron
        for neuron_idx1 in range(cts.MAP_SIZE):
            for neuron_idx2 in range(cts.MAP_SIZE):
                if (neuron_idx1, neuron_idx2) == (0, 0):
                    continue
                output = np.dot(weights[neuron_idx1, neuron_idx2].transpose(), input_vector)
                if output > max_output:
                    max_output = output
                    winning_neuron_idx = (neuron_idx1, neuron_idx2)
        test_outcome[winning_neuron_idx] = output_dict[data_idx]

    print 'Output of the network:'
    for neuron_idx1 in range(cts.MAP_SIZE):
        output_line = ''
        for neuron_idx2 in range(cts.MAP_SIZE):
            neuron_idx = (neuron_idx1, neuron_idx2)
            if neuron_idx in test_outcome.keys():
                output_line += '\t' + '{:15s}'.format(test_outcome[neuron_idx])
            else:
                output_line += '\t' + '{:15s}'.format(' - ')
        print output_line
        print ''


