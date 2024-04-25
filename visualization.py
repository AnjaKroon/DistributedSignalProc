
import matplotlib.pyplot as plt
from math import log

'''
PLOTTING CONVERGENCE TIME e(k)
''' 
def plot_single_error(array, name):
    x_values = [x for x, _ in array]
    y_values = [y for _, y in array]
    plt.plot(x_values, y_values)
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title('e(k) vs. Trans. ' + name, fontweight='bold')
    plt.yscale('log') 
    plt.legend(name)
    plt.show()

def plot_multiple_pairs(pairs):
    for array, name in pairs:
        x_values = [x for x, _ in array]
        y_values = [y for _, y in array]
        plt.plot(x_values, y_values, label=name)
    plt.xlabel('Transmissions')
    plt.ylabel('||x_k - x_avg||^2')
    plt.title('e(k) vs. Trans.')
    plt.yscale('log') 
    plt.legend([name for _, name in pairs])
    plt.show()

def plot_c_transmissions(array, name, tolerance):
    x_values = [x for x, _ in array]
    y_values = [y for _, y in array]
    plt.plot(x_values, y_values)
    plt.xlabel('c')
    plt.ylabel('Transmissions')
    plt.title('Choice of c for ' + name + ". Tolerance: " + str(tolerance), fontweight='bold')
    plt.legend(name)
    plt.show()


