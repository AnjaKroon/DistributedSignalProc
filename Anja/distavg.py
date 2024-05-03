
import networkx as nx
import numpy as np
import random
import statistics
import time

from utils import generate_measurements, generate_rgg, vector_to_dict
from visualization import plot_rgg_side_by_side

'''
SYNCH. DIST AVG
'''
def dist_avg_synch(graph, TOL):
    '''
    1) Find optimal alpha
    2) Set up W matrix
    3) Set x(k-1) = x(0) for algorithm start
    4) While e(k) > epsilon:
        x(k) = W*x(k-1)
        e(k) = || . ||
        x(k-1) = x(k)
    TRANSMISSIONS: for each iteration, transmissions increase by the number of edges in the graph
    Notice: W works on every entry in x(0) at each iteration, every link is active at each iteration -- confirm numerically
    '''
    print("")
    print("------- SYNCH. DIST. AVG. ------- ")
    start_time = time.time()
    
    # Find optimal alpha
    all_nodes = list(nx.nodes(graph))

    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues_list = eigenvalues.tolist()

    eigenvalues_list.sort(reverse=True)                 # sort in decending order to get eig1 and eigN-1
    alpha_opt = 2 / (eigenvalues_list[0] + eigenvalues_list[-2])
    
    # Set up W matrix
    W = np.zeros((len(all_nodes), len(all_nodes)))
    W = np.eye(len(all_nodes)) - alpha_opt * laplacian

    num_edges = np.count_nonzero(np.triu(W, k=1))

    row_sums = np.sum(W, axis=1)
    col_sums = np.sum(W, axis=0)
    if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1):
        print("\033[91mERROR: The sum of rows or columns in W does not equal 1.\033[0m")
    elif not np.allclose(W, W.T):
        print("\033[91mERROR: W is not symmetric.\033[0m")

    # Set x(k-1) = x(0) for algorithm start
    x_kminus1 = np.array(list(nx.get_node_attributes(graph, "temp").values()))   

    # get required true average needed for the e(k) function   
    true_avg = np.mean(x_kminus1)   

    # initializations for the while loop                                               
    x_k = np.zeros(len(all_nodes))
    transmissions = 0
    std_devs = []
    errors = []

    # while num_iterations < 25 or num_iterations > 20000 or not np.allclose(x_kminus1, true_avg):
    while (np.linalg.norm(x_k - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        x_k = np.dot(W, x_kminus1)                          # update the x vector
        std_devs.append((transmissions, np.std(x_k)))
        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(x_k - np.ones(len(all_nodes)) * true_avg)**2))
        # x(k-1) = x(k)
        x_kminus1 = x_k

        # TRANSMISSIONS: for each iteration, transmissions increase by the number of edges in the graph
        transmissions += num_edges

    asym_conv_factor = np.max(np.linalg.eigvals(W - np.ones((len(all_nodes), len(all_nodes))) / len(all_nodes)))
    # print("Asymptotic convergence factor", asym_conv_factor)
    
    if np.max(np.mean(x_kminus1 - true_avg)) > TOL:
        print(f"\033[91mERROR: Not all values in x_k are within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x_kminus1[0])
    return x_kminus1[0], std_devs, errors, transmissions

'''
ASYNCH DIST AVG
'''
def dist_avg_asynch_W(graph, TOL):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        find all neighbors of node i
        Two methods to average:
            ''manual averaging of i and neighbor i values''
                x(k) = avg(x_i U x_ni)
            ''using the W matrix'' -- USED HERE
                construct W matrix
                check that W matrix obeys rules
                x(k) = W(k)x(k-1)
                x(k-1) = x(k)
        e(k) = || . ||
    TRANSMISSIONS: per iteration of while loop = N(i)
    '''
    print("")
    print("------- ASYNCH. DIST. AVG. (W)------- ")
    start_time = time.time()

    # x(k-1) = x(0)
    x_kminus1 = np.array(list(nx.get_node_attributes(graph, "temp").values())) 

    # setup for e(k)
    true_avg = np.mean(x_kminus1)

    # initializations for the while loop
    all_nodes = list(nx.nodes(graph))  
    x_k = np.zeros(len(all_nodes))
    transmissions = 0 
    std_devs = []
    errors = []

    # while e(k) > epsilon:
    while (np.linalg.norm(x_k - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        # uniformly select random node i
        random_node_i = random.choice(all_nodes)

        # Construct the W matrix according to the current node selected
        # find all neighbors of node i
        list_neighbors_cur = list(nx.all_neighbors(graph, random_node_i))
        list_neighbors_cur.append(random_node_i)

        # Calculate the W matrix for node i
        W = np.zeros((len(all_nodes), len(all_nodes))) 

        for l in list_neighbors_cur:
            for m in list_neighbors_cur:
                W[l, m] = 1 / (graph.degree(random_node_i) + 1)
        
        for l in range(len(all_nodes)):
            for m in range(len(all_nodes)):
                if l == m and l not in list_neighbors_cur:
                    W[l, m] = 1
        
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
        
        # x(k) = W(k)x(k-1)
        x_k = np.dot(W, x_kminus1)
        std_devs.append((transmissions, np.std(x_k)))
        errors.append((transmissions, np.linalg.norm(x_k - np.ones(len(all_nodes)) * true_avg)**2))
        x_kminus1 = x_k

        # TRANSMISSIONS: per iteration of while loop = N(i)
        transmissions += len(list_neighbors_cur)

    if np.max(np.mean(x_kminus1 - true_avg)) > TOL:
        print(f"\033[91mERROR: Not all values in x_k are within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x_kminus1[0])

    return x_kminus1[0], std_devs, errors, transmissions

'''
ASYNCH DIST AVG
'''
def dist_avg_asynch_noW(graph, TOL):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        find all neighbors of node i
        Two methods to average:
            ''manual averaging of i and neighbor i values'' -- USED HERE
                x(k) = avg(x_i U x_ni)
                e(k) = || . ||
            ''using the W matrix'' 
                construct W matrix
                check that W matrix obeys rules
                x(k) = W(k)x(k-1)
                e(k) = || . ||
                x(k-1) = x(k)
    TRANSMISSIONS: per iteration of while loop = N(i)
    '''
    print("")
    print("------- ASYNCH. DIST. AVG. (no W)------- ")

    start_time = time.time()

    # x(k-1) = x(0)
    all_temps = nx.get_node_attributes(graph, "temp")
    
    # get true average, used for stopping criterion
    true_avg = np.mean(list(all_temps.values()))
    std_devs = []
    errors = []
    transmissions = 0

    all_nodes = list(nx.nodes(graph))
    while (np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # find all neighbors of node i
        neighbors_i = list(nx.all_neighbors(graph, node_i))

        # before computing the average, get values in set N(i) U i
        cur_temps = [all_temps[node] for node in neighbors_i]
        cur_temps.append(all_temps[node_i])

        avg = sum(cur_temps) / len(cur_temps)

        # update all nodes in set N(i) U i
        for node in neighbors_i:
            all_temps[node] = avg
        all_temps[node_i] = avg

        # TRANSMISSIONS: per iteration of while loop = N(i)
        transmissions += len(neighbors_i)

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions

'''
ASYNCH DIST AVG TF
'''
def dist_avg_asynch_noW_tf(graph, TOL, FAILURE_RATE=0.0):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        find all neighbors of node i
            x(k) = avg(x_i U x_ni)
            e(k) = || . ||
    TRANSMISSIONS: per iteration of while loop = N(i)
    '''
    print("")
    print("------- ASYNCH. DIST. AVG. UNDER TF------- ")

    start_time = time.time()

    # x(k-1) = x(0)
    all_temps = nx.get_node_attributes(graph, "temp")
    
    # get true average, used for stopping criterion
    true_avg = np.mean(list(all_temps.values()))
    std_devs = []
    errors = []
    transmissions = 0

    all_nodes = list(nx.nodes(graph))
    while (np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        # generate a random number between 0 and 1
        random_number = random.randint(0, 1)

        if transmissions > 80000: # limit the number of transmissions
            break

        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # find all neighbors of node i
        neighbors_i = list(nx.all_neighbors(graph, node_i))

        # TF: does not see neighbors which are there
        # do some failures here
        if random_number == 1 and FAILURE_RATE >0:
            num_failures = int(len(neighbors_i) * FAILURE_RATE)
            neighbors_i = random.sample(neighbors_i, len(neighbors_i) - num_failures)

        # before computing the average, get values in set N(i) U i
        cur_temps = [all_temps[node] for node in neighbors_i]
        cur_temps.append(all_temps[node_i])

        avg = sum(cur_temps) / len(cur_temps)

        # update all nodes in set N(i) U i
        # TF Can occur here, potentially not reaching all nodes in the neighborhood
        if random_number == 0 and FAILURE_RATE >0:
            num_failures = int(len(neighbors_i) * FAILURE_RATE)
            neighbors_i = random.sample(neighbors_i, len(neighbors_i) - num_failures)

        for node in neighbors_i:
            all_temps[node] = avg
        all_temps[node_i] = avg

        # TRANSMISSIONS: per iteration of while loop = N(i)
        transmissions += len(neighbors_i)

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2))

    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions

'''
ASYNCH DIST AVG DROP/ADD
'''
def dist_avg_asynch_noW_dropadd(graph, TOL, DROP_RATE=0.0, ADD_RATE=0.0, type="bulk"):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        find all neighbors of node i
        Two methods to average:
            ''manual averaging of i and neighbor i values'' -- USED HERE
                x(k) = avg(x_i U x_ni)
                e(k) = || . ||
    TRANSMISSIONS: per iteration of while loop = N(i)
    '''
    print("")
    if (DROP_RATE > 0.0): # Dropping nodes
        print("------- DIST AVG w/ " + str(type) + " DROP: " + str(DROP_RATE) + " ------- ")
    elif (ADD_RATE > 0.0): # Adding nodes
        print("------- DIST AVG w/ " + str(type) + " ADD: " + str(ADD_RATE) + " ------- ")
    else: print("------- DIST AVG w/ DROP: " + str(DROP_RATE) + " ADD " + str(ADD_RATE) + " ------- ")

    start_time = time.time()

    # x(k-1) = x(0)
    all_temps = nx.get_node_attributes(graph, "temp")
    
    # get true average, used for stopping criterion
    true_avg = np.mean(list(all_temps.values()))
    std_devs = []
    errors = []
    transmissions = 0

    DROPPED_FLAG = False
    ADDED_FLAG = False

    all_nodes = list(nx.nodes(graph))

    num_nodes_drop = int(len(all_nodes) * DROP_RATE) # in total this is how many nodes you want to drop

    num_nodes_add = int(len(all_nodes) * ADD_RATE) # in total this is how many nodes you want to add
    print("Number of nodes to add: ", num_nodes_add)

    while (np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        if transmissions > 100000:
            break
        # CASE 1: BULK DROP
        if (transmissions > 10000 and DROPPED_FLAG == False and type == "bulk" and DROP_RATE > 0.0):
            graph_old = graph.copy()
            num_nodes_drop = int(len(all_nodes) * DROP_RATE)
            nodes_to_drop = random.sample(all_nodes, num_nodes_drop)
            for node in nodes_to_drop:
                graph.remove_node(node)
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been dropped in bulk, the graph is no longer connected.\033[0m")
                break

            plot_rgg_side_by_side(graph_old, graph, "Bulk Drop")
            
            # Recalculation after nodes drop
            all_nodes = list(nx.nodes(graph))
            all_temps = {node: temp for node, temp in all_temps.items() if node in graph}
            # true_avg = np.mean(list(all_temps.values()))
            DROPPED_FLAG = True
        
        # CASE 2: BULK ADD
        if (transmissions > 10000 and ADDED_FLAG == False and type == "bulk" and ADD_RATE > 0.0):
            graph_old = graph.copy()
            print("Number of nodes BEFORE BULK ADD: ", len(all_nodes))
            num_nodes_add = int(len(all_nodes) * ADD_RATE)
            new_measurements = generate_measurements(num_nodes_add)

            for i in range(num_nodes_add):
                # print("Node to add: ", len(all_nodes)+i+1)
                # is position supposed to be in this range
                graph.add_node(len(all_nodes)+1+i, pos=(random.uniform(0, 1), random.uniform(0, 1)), temp=new_measurements[i])
                # make sure graph position is being created as well

                pos = nx.get_node_attributes(graph, 'pos') 
                
                # print(len(pos), "should be equal to ", len(all_nodes)+i+1)

                # randomly connect it to the graph in a random geometric manner
                RADIUS = 0.12
                
                for node in graph.nodes():
                    if node != len(all_nodes) + i + 1 and node not in all_nodes[-num_nodes_add:]:
                        distance = np.linalg.norm(np.array(graph.nodes[node]['pos']) - np.array(graph.nodes[len(all_nodes) + i + 1]['pos']))
                        if distance <= RADIUS:
                            if not graph.has_edge(node, len(all_nodes) + i + 1):
                                graph.add_edge(node, len(all_nodes) + i + 1)

            # to check this was done correctly, print the graph before and after adding nodes
            plot_rgg_side_by_side(graph_old, graph, "Bulk Add")

            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been added in bulk, the graph is no longer connected.\033[0m")
                break
            
            # Recalculation after nodes drop
            all_nodes = list(nx.nodes(graph))
            print("Number of nodes AFTER BULK ADD: ", len(all_nodes))
            # all_temps = nx.get_node_attributes(graph, "temp")
            all_temps.update({node: graph.nodes[node]['temp'] for node in all_nodes[-num_nodes_add:]})
            # true_avg = np.mean(list(all_temps.values()))
            
            ADDED_FLAG = True

        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # find all neighbors of node i
        neighbors_i = list(nx.all_neighbors(graph, node_i))

        # before computing the average, get values in set N(i) U i
        cur_temps = [all_temps[node] for node in neighbors_i]
        cur_temps.append(all_temps[node_i])

        avg = sum(cur_temps) / len(cur_temps)

        # update all nodes in set N(i) U i
        for node in neighbors_i:
            all_temps[node] = avg
        all_temps[node_i] = avg

        # TRANSMISSIONS: per iteration of while loop = N(i)
        transmissions += len(neighbors_i)

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2))

    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", list(all_temps.values())[0])

    return list(all_temps.values())[0], std_devs, errors, transmissions

