# Code for distributed SP project
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

# GENERATE RANDOM GRAPH
def generate_rgg(num_nodes, radius, dimen, meas):
    # check if graph connected
    is_connected = False
    while(not is_connected):
        rgg = nx.random_geometric_graph(num_nodes, radius, dim=dimen, p=2)
        # check if connected
        is_connected = nx.is_connected(rgg)

    pos = nx.get_node_attributes(rgg, 'pos') #positions
    nx.draw(rgg, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8)

    plt.axis('equal')
    plt.xlabel('100 km^2')
    plt.ylabel('100 km^2')
    plt.title("test")
    plt.show()

    nx.set_node_attributes(rgg, meas, "temp")

    return rgg

# GENERATE RANDOM MEASUREMENTS
def generate_measurements(num_nodes):
    n = np.random.normal(0, 3, num_nodes)
    t = 20
    y = np.zeros(num_nodes)
    for i in range(len(n)):
         y[i] = t + n[i]
    return y

def vector_to_dict(vector):
    return {index: value for index, value in enumerate(vector)}

# Implementation of Distributed averaging
def dist_avg(graph):
    std_dev = 100
    all_temps = nx.get_node_attributes(graph, "temp")
    std_devs = []
    while(std_dev > 0.0001):
        all_nodes = list(nx.nodes(graph))
        # print(all_nodes)
        cur_node = random.choice(all_nodes)
        # print(cur_node)
        list_neighbors_cur = list(nx.all_neighbors(graph, cur_node))
        # print(list_neighbors_cur)

        rand_neigh = random.choice(list_neighbors_cur)

        # exchange information with neighbor
        # get current node temperature, get random node, average them both, 
        # reset cur and neighbor node temperature
        cur_temp = all_temps[cur_node]
        neigh_temp = all_temps[rand_neigh]

        avg = (cur_temp + neigh_temp)/2
        all_temps.update({cur_node: avg, rand_neigh: avg})

        # report standard dev of all node temps for each iteration
        std_dev = statistics.stdev(list(all_temps.values()))

        std_devs.append(std_dev)
        # stop when the standard dev is less than 0.00001
    print("Successfully converged to a solution")
    return std_devs


def plot_std_devs(array):
    x_values = range(len(array))
    print("Number of transmissions required for convergence with randomized gossip: ", len(array))
    plt.plot(x_values, array, marker='o', linestyle='-')
    plt.xlabel('Transmissions')
    plt.ylabel('Measurement Std Dev')
    plt.title('Measurement Std Dev vs. Transmissions')
    plt.show()
    

# IMPLEMENT OTHER METHOD FOR SOLVING

def main():
    nodes = 32
    rad = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01
    dim = 2
    # Generate measurements and random graph
    temps = generate_measurements(nodes)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(nodes, rad, dim, dict_temps)

    standard_deviations = dist_avg(rand_geo_gr)
    plot_std_devs(standard_deviations)




if __name__ == "__main__":
    main()
