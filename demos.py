import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def mi_ex():
    plt.close()
    fuzzed_sin = np.sin(np.arange(450) * 2 * np.pi / 360) + npr.normal(scale=0.1, size=450) + 1
    plt.scatter(np.arange(450), fuzzed_sin)
    plt.savefig("mi_ex")

if __name__ == "__main__":
    mi_ex()
