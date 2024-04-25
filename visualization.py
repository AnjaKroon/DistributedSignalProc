
import matplotlib.pyplot as plt
from math import log

'''
PLOTTING CONVERGENCE TIME e(k)
''' 
def plot_single_error(array, name):
    x_values = [x for x, _ in array]
    y_values = [y for _, y in array]
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title('e(k) vs. Trans. ' + name, fontweight='bold')
    plt.yscale('log') 
    plt.show()

def plot_multiple_pairs(pairs):
    for array, name in pairs:
        x_values = [x for x, _ in array]
        y_values = [y for _, y in array]
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=name)
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title('e(k) vs. Trans.')
    plt.yscale('log') 
    plt.legend()
    plt.show()
