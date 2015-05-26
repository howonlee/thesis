import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sci_stats

def mi_ex():
    plt.close()
    sin_size = 180
    fuzzed_sin = np.sin(np.arange(sin_size) * 2 * np.pi / 360) + npr.normal(scale=0.1, size=sin_size) + 1
    plt.scatter(np.arange(sin_size), fuzzed_sin)
    print sci_stats.stats.pearsonr(range(fuzzed_sin.size), fuzzed_sin)
    print cross_mutual_information(range(fuzzed_sin.size), fuzzed_sin, 1, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("mi_ex")

def prob_dict(data):
    data = list(data)
    alphabet = set(data)
    probdict = {}
    for symbol in alphabet:
        ctr = sum(1 for x in data if x == symbol)
        probdict[symbol] = (float(ctr) / len(data))
    return probdict

def entropy(data):
    data_probs = prob_dict(data)
    return sum(-p * np.log2(p) for _, p in data_probs.iteritems())

def joint_entropy(data1, data2):
    probs = []
    data1, data2 = np.array(data1, dtype=np.int16), np.array(data2, dtype=np.int16)
    #should look at the sparsity of things
    for c1 in set(data1):
        for c2 in set(data2):
            probs.append(np.mean(np.logical_and(data1 == c1, data2 == c2)))
    probs = filter(lambda x: x != 0.0, probs)
    return np.sum(-p * np.log2(p) for p in probs)

def mutual_information(data1, data2):
    data1, data2 = list(data1), list(data2)
    return entropy(data1) + entropy(data2) - joint_entropy(data1, data2)

def cross_mutual_information(data1, data2, stepsize, stepmax):
    first = np.array(data1)
    cmis = []
    for lag in xrange(0, stepmax, stepsize):
        lagged = np.roll(data2, -lag)
        cmis.append(mutual_information(first, lagged))
    return cmis

if __name__ == "__main__":
    mi_ex()
