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
import time

NODES = 1000
RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01
DIM = 2
TOL = 0.001

'''
GENERATING RANDOM GEOMETRIC GRAPH
using generate_rgg(), generate_measurements(), vector_to_dict
'''
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



'''
SYNCH. DIST AVG
Implementation of the dist avg algorithm with W matrix -- around slide 22 of distributed lecture
Construct W, implement the linear iterations step, check it converges to the average
enforced that W is symmetric -- optimal symmetric weights -- fastest convergence rate
'''
def dist_avg_withW(graph):
    print("")
    print("------- SYNCH. DIST. AVG. ------- ")
    start_time = time.time()

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
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    
    return x_kminus1[0], std_devs

'''
ASYNCH DIST AVG
Should be: less sensitive to changes in network topology than synch., less execution time than synch.
How works:
kth time slot, select node i uniform randomly, let it contact its neighboring nodes,
and set their values equal to the average of the current values
Draw the matricies W(k) iid from some distribution
If picking W whose mean satiesfies the spectral radius condition, will converge to soln.
'''
def dist_avg_asynch(graph):
    print("")
    print("------- ASYNCH. DIST. AVG. ------- ")
    start_time = time.time()

    # Initializations
    x_kminus1 = np.array(list(nx.get_node_attributes(graph, "temp").values())) 
    true_avg = np.mean(x_kminus1)
    all_nodes = list(nx.nodes(graph))  
    x_k = np.zeros(len(all_nodes))
    num_iterations = 0  
    std_devs = []

    # repeat this until convergence
    while num_iterations < 25 or num_iterations > 20000 or not np.allclose(x_kminus1, true_avg):
        # Select node i at random
        random_node_i = random.choice(all_nodes)

        # Construct the W matrix according to the current node selected
        list_neighbors_cur = list(nx.all_neighbors(graph, random_node_i))
        list_neighbors_cur.append(random_node_i)

        # Calculate the W matrix for node i -- effectively sampling from distribution of W because i is random
        W = np.zeros((len(all_nodes), len(all_nodes))) 

        for l in list_neighbors_cur:
            for m in list_neighbors_cur:
                W[l, m] = 1 / (graph.degree(random_node_i) + 1)
        
        for l in range(len(all_nodes)):
            for m in range(len(all_nodes)):
                if l == m and l not in list_neighbors_cur:
                    W[l, m] = 1
        # print("W")
        # print('\n'.join(['\t'.join([str(round(cell, 3)) for cell in row]) for row in W]))

        # Round the values in the W matrix to be two more decimals than the sig figs in number of nodes
        # num_decimals = int(np.ceil(np.log10(NODES))) + 5
        # W = np.round(W, decimals=num_decimals)
        # to help prevent overflow
        
        # Check if W is symmetric
        if not np.allclose(W, W.T):
            print("\033[91mERROR: W is not symmetric.\033[0m")
            break
        
        # Check if W is doubly stochastic
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1):
            print("\033[91mERROR: W is not doubly stochastic.\033[0m")
            break

        # Check if W is nonnegative
        if not np.all(W >= 0):
            print("\033[91mERROR: W is not nonnegative.\033[0m")
            break

        # Check if abs(eigenvalues(W)) <= 1 + TOL (must do tolerance to account for overflow)
        eigenvalues = np.linalg.eigvals(W)
        if not np.all(np.abs(eigenvalues) <= 1.0 + TOL):
            print("\033[91mERROR: Absolute value of eigenvalues of W is not less than or equal to 1.\033[0m")
            break
        
        # If W has been calculated correctly, calculate x(k)
        x_k = np.dot(W, x_kminus1)
        # print("Mean:", np.mean(x_k), "Std dev: ", np.std(x_k))
        std_devs.append(np.std(x_k))
        x_kminus1 = x_k

        num_iterations += 1

    if np.max(np.abs(x_kminus1 - true_avg)) > TOL:
        print(f"\033[91mERROR: Not all values in x_k are within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    # should be lining up to slide 45
    
    return x_kminus1[0], std_devs


    # Things to check
    # W should be symmetric
    # W should be doubly stochastic
    # W should be nonnegative
    # absolute value of all eig values of W at all time should be less than or equal to 1


'''
RANDOMIZED GOSSIP
no global knowledge of network, only contact one neighbor at a time
Implementation of randomized gossip? -- randomly selecting neighbor and averaging
exchange information with neighbor
get current node temperature, get random node, average them both, 
reset cur and neighbor node temperature
'''
def random_gossip(graph):
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

'''
PLOTTING CONVERGENCE TIME
using the standard deviations -- mean did not provide sufficient information
'''
def plot_single_std_dev(array, name):
    x_values = range(len(array))
    print("Messages for " + name + ":  ", len(array))
    plt.plot(x_values, array, marker='o', linestyle='-')
    plt.xlabel('Messages')
    plt.ylabel('Measurement Std Dev')
    plt.title('X_avg Std Dev vs. Trans. ' + name, fontweight='bold')
    plt.yscale('log') 
    plt.show()

def plot_two_std_devs(array1, array2, name1, name2):
    x_values = range(max(len(array1), len(array2)))

    # pad the shorter one with None so they can be plotted together
    if len(array1) < len(array2):
        array1 += [None] * (len(array2) - len(array1))
    elif len(array2) < len(array1):
        array2 += [None] * (len(array1) - len(array2))

    print("Messages for " + name1 + ":  ", len(array1))
    print("Messages for " + name2 + ":  ", len(array2))
    plt.plot(x_values, array1, marker='o', linestyle='-', label=name1)
    plt.plot(x_values, array2, marker='o', linestyle='-', label=name2)
    plt.xlabel('Messages')
    plt.ylabel('Measurement Std Dev')
    plt.title('X_avg Std Dev vs. Trans.')
    plt.yscale('log') 
    plt.legend()
    plt.show()

'''
PLOTTING CONVERGENCE TIME WITH e(k)
''' 


def main():

    temps = generate_measurements(NODES)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(NODES, RAD, DIM, dict_temps)

    '''
    avg_rand_gossip, stdev_rand_gossip = random_gossip(rand_geo_gr)
    plot_std_devs(stdev_rand_gossip, "Rand Gossip (Message Passing)")
    print("Average with Rand Gossip (Message Passing)", avg_rand_gossip)
    '''

    # SYNCH DIST AVG
    avg_withW, stdev_dist_avg_withW = dist_avg_withW(rand_geo_gr)
    # plot_single_std_dev(stdev_dist_avg_withW, "Synch. Dist. Avg.")
    print("Average with Synch. Dist. Avg. ", avg_withW)

    # ASYNCH DIST AVG
    avg_dist_avg_asynch, stdev_dist_avg_asynch = dist_avg_asynch(rand_geo_gr)
    # plot_single_std_dev(stdev_dist_avg_asynch, "Asynch. Dist. Avg.")
    print("Average with Asynch. Dist. Avg.", avg_dist_avg_asynch)

    plot_two_std_devs(stdev_dist_avg_withW, stdev_dist_avg_asynch, "Synch. Dist. Avg.", "Asynch. Dist. Avg.")

if __name__ == "__main__":
    main()
