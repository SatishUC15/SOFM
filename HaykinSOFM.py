import numpy as np
import random as rand
import Constants as cts
from scipy.spatial.distance import euclidean
import SOFM

weights = {}

# Generating the data for representing 16 animals using attribute code and symbol code
attribute_code = np.array(
                    np.matrix(  '1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0;'
                                '0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0;'
                                '0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1;'
                                '1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0;'
                                '0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1;'
                                '0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1;'
                                '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1;'
                                '0 0 0 1 0 0 0 0 0 1 0 0 1 1 1 0;'
                                '1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0;'
                                '0 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0;'
                                '0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 0;'
                                '1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0;'
                                '0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0').transpose())

symbol_code = []
for idx1 in range(len(cts.ANIMALS)):
    temp = []
    for idx2 in range(len(cts.ANIMALS)):
        if idx2 == idx1:
            temp.append(cts.CONSTANT)
        else:
            temp.append(0)
    symbol_code.append(temp)
symbol_code = np.array(symbol_code)

# Initializing small random weights for the network
for idx1 in range(cts.MAP_SIZE):
    for idx2 in range(cts.MAP_SIZE):
        temp = []
        for idx3 in range(len(cts.ANIMALS) + cts.NUM_ATTRIBUTES):
            temp.append(rand.random() / 100)
        weights[idx1, idx2] = temp


# Train the SOFM with the existing data
updated_weight_matrix = SOFM.train_sofm(weights, symbol_code, attribute_code)
print "Weights obtained:", updated_weight_matrix

test_attribute_code = np.zeros((len(cts.ANIMALS), 13))

# Test the SOFM model with a zero attribute vector
SOFM.test_sofm(updated_weight_matrix, symbol_code, test_attribute_code, 'test')

# Testing the SOFM model with new input and zero symbol_code
new_symbol_codes = np.zeros((len(cts.NEW_ANIMALS), 16))

new_attribute_codes = np.array(np.matrix(  '0 1 0 0 1 1 1 0 0 0 0 0 0;'
                                           '0 0 1 0 1 1 1 0 0 0 0 0 0;'
                                           '1 0 0 0 1 1 0 0 0 1 0 0 0;'
                                           '0 0 1 1 0 0 0 0 1 0 1 0 0;'
                                           '1 0 0 1 0 1 0 0 0 1 0 1 0;'
                                           '0 0 1 0 0 0 0 0 0 0 0 0 1;'
                                           '0 0 1 0 0 0 0 0 0 1 0 0 1'))

SOFM.test_sofm(updated_weight_matrix, new_symbol_codes, new_attribute_codes, 'new')

max_similarity = {}
for idx1 in range(len(cts.NEW_ANIMALS)):
    min_idx = 0
    min_dist = euclidean(new_attribute_codes[idx1], attribute_code[0])
    for idx2 in range(1, len(cts.ANIMALS)):
        dist = euclidean(new_attribute_codes[idx1], attribute_code[idx2])
        if dist < min_dist:
            min_dist = dist
            min_idx = idx2
    max_similarity[cts.NEW_ANIMALS[idx1]] = cts.ANIMALS[min_idx]
print "Similarity of New Input to existing Input:\n", max_similarity
