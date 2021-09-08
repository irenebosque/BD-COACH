import numpy as np

def loadWeights(path, weights_name, test_numbers):
    tests = []
    for number in test_numbers:

        test = np.load(path + weights_name + number + ".npy", allow_pickle=True)
        tests.append(test)

    return tests

