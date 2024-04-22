# AUTHOR: Anja Kroon

# Code for distributed SP project
# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
from math import log

NODES = 100
RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01
DIM = 2
TOL = 0.001

def generate_rgg(num_nodes, radius, dimen, meas):
    is_connected = False
    while(not is_connected):
        rgg = nx.random_geometric_graph(num_nodes, radius, dim=dimen, p=2)
        is_connected = nx.is_connected(rgg)

    pos = nx.get_node_attributes(rgg, 'pos') 
    nx.draw(rgg, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8)

    plt.axis('equal')
    plt.xlabel('100 km^2')
    plt.ylabel('100 km^2')
    plt.title("test")
    plt.show()

    nx.set_node_attributes(rgg, meas, "temp")

    return rgg

def generate_measurements(num_nodes):
    n = np.random.normal(0, 3, num_nodes)
    t = 20
    y = np.zeros(num_nodes)
    for i in range(len(n)):
         y[i] = t + n[i]
    return y

def vector_to_dict(vector):
    return {index: value for index, value in enumerate(vector)}

# Implementation of Distributed averaging -- randomly selecting neighbor and averaging
# exchange information with neighbor
# get current node temperature, get random node, average them both, 
# reset cur and neighbor node temperature
def dist_avg(graph):
    std_dev = 100
    all_temps = nx.get_node_attributes(graph, "temp")
    x_avg = np.mean(list(all_temps.values()))
    std_devs = []
    while(std_dev > 0.01 or np.max(np.abs(list(all_temps.values()) - np.mean(list(all_temps.values())))) > 0.01):
        all_nodes = list(nx.nodes(graph))
        cur_node = random.choice(all_nodes)

        list_neighbors_cur = list(nx.all_neighbors(graph, cur_node))
        rand_neigh = random.choice(list_neighbors_cur)

        cur_temp = all_temps[cur_node]
        neigh_temp = all_temps[rand_neigh]

        avg = (cur_temp + neigh_temp)/2
        all_temps.update({cur_node: avg, rand_neigh: avg})

        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append(std_dev)

    if np.max(np.mean(list(all_temps.values())) - x_avg) > TOL:
        print(f"\033[91mERROR: Not all values in x_k are within {TOL} of each other.\033[0m")

    return all_temps[0], std_devs

# Implementation of the dist avg algorithm with W matrix -- around slide 22 of distributed lecture
# Construct W, implement the linear iterations step, check it converges to the average
# enforced that W is symmetric -- optimal symmetric weights -- fastest convergence rate
# synchronous distributed averaging?
def dist_avg_withW(graph):
    all_nodes = list(nx.nodes(graph))

    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues_list = eigenvalues.tolist()

    eigenvalues_list.sort(reverse=True)                 # sort in decending order to get eig1 and eigN-1
    alpha_opt = 2 / (eigenvalues_list[0] + eigenvalues_list[-2])
    
    W = np.zeros((len(all_nodes), len(all_nodes)))
    W = np.eye(len(all_nodes)) - alpha_opt * laplacian

    row_sums = np.sum(W, axis=1)
    col_sums = np.sum(W, axis=0)
    if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1):
        print("\033[91mERROR: The sum of rows or columns in W does not equal 1.\033[0m")
    elif not np.allclose(W, W.T):
        print("\033[91mERROR: W is not symmetric.\033[0m")

    x_kminus1 = np.array(list(nx.get_node_attributes(graph, "temp").values()))      # initialize x(k-1)
    true_avg = np.mean(x_kminus1)                                                   # get x_avg vector
    x_k = np.zeros(len(all_nodes))
    num_iterations = 0
    std_devs = []

    while num_iterations < 25 or num_iterations > 20000 or not np.allclose(x_kminus1, true_avg):
        x_k = np.dot(W, x_kminus1)                          # update the x vector
        std_devs.append(np.std(x_k))
        
        x_kminus1 = x_k
        num_iterations += 1
    
    asym_conv_factor = np.max(np.linalg.eigvals(W - np.ones((len(all_nodes), len(all_nodes))) / len(all_nodes)))
    print("Asymptotic convergence factor", asym_conv_factor)
    
    if np.max(np.abs(x_kminus1 - true_avg)) > TOL:
        print(f"\033[91mERROR: Not all values in x_k are within {TOL} of each other.\033[0m")
    
    return x_kminus1[0], std_devs

def plot_std_devs(array, name):
    x_values = range(len(array))
    print("Messages for " + name + ":  ", len(array))
    plt.plot(x_values, array, marker='o', linestyle='-')
    plt.xlabel('Messages')
    plt.ylabel('Measurement Std Dev')
    plt.title('X_avg Std Dev vs. Trans. ' + name, fontweight='bold')
    plt.yscale('log') 

    plt.show()
    

def main():

    temps = generate_measurements(NODES)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(NODES, RAD, DIM, dict_temps)

    '''
    avg, stdev_dist_avg = dist_avg(rand_geo_gr)
    plot_std_devs(stdev_dist_avg, "Dist. Avg. (Message Passing)")
    print("Average with Dist. Avg. (Message Passing)", avg)
    '''

    # SYNCH DIST AVG WITH W
    avg_withW, stdev_dist_avg_withW = dist_avg_withW(rand_geo_gr)
    plot_std_devs(stdev_dist_avg_withW, "Synch. Dist. Avg. (With W)")
    print("Average with Synch. Dist. Avg. (With W)", avg_withW)


if __name__ == "__main__":
    main()
