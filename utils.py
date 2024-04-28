
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

'''
GENERATING RANDOM GEOMETRIC GRAPH
using generate_rgg(), generate_measurements(), vector_to_dict
'''
def generate_measurements(num_nodes):
    n = np.random.normal(0, 3, num_nodes)
    t = 20
    y = np.zeros(num_nodes)
    for i in range(len(n)):
         y[i] = t + n[i]
    return y

def generate_rgg(num_nodes, radius, dimen, meas):
    # assuming a connected graph -- do we need to consider non connected graphs?
    is_connected = False
    while(not is_connected):
        rgg = nx.random_geometric_graph(num_nodes, radius, dim=dimen, p=2)
        is_connected = nx.is_connected(rgg)

    pos = nx.get_node_attributes(rgg, 'pos') 
    nx.draw(rgg, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8)

    plt.axis('equal')
    plt.xlabel('100 km^2')
    plt.ylabel('100 km^2')
    plt.title("Graph of " + str(num_nodes) + " nodes with radius " + str(radius) + " and dimension " + str(dimen))
    plt.show()

    nx.set_node_attributes(rgg, meas, "temp")

    return rgg


def vector_to_dict(vector):
    return {index: value for index, value in enumerate(vector)}

