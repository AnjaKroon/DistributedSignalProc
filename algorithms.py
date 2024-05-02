
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

    if np.max(np.mean(list(all_temps.values())) - true_avg) > TOL:
        print(f"\033[91mERROR: On average the values in x_k are not within {TOL} of each other.\033[0m")
    
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
        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # find all neighbors of node i
        neighbors_i = list(nx.all_neighbors(graph, node_i))

        # TF: does not see neighbors which are there
        # do some failures here
        # randomly drop out transmissions with failure_rate amount of neighbors
        num_failures = int(len(neighbors_i) * 0.5*FAILURE_RATE)
        neighbors_i = random.sample(neighbors_i, len(neighbors_i) - num_failures)

        # before computing the average, get values in set N(i) U i
        cur_temps = [all_temps[node] for node in neighbors_i]
        # TF: does not get all the averages back

        # do some failures here
        # randomly drop out some pairs in cur_temps
        num_dropouts = int(len(cur_temps) * 0.5 * FAILURE_RATE)
        cur_temps = random.sample(cur_temps, len(cur_temps) - num_dropouts)
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

    if np.max(np.mean(list(all_temps.values())) - true_avg) > TOL:
        print(f"\033[91mERROR: On average the values in x_k are not within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions


'''
RANDOMIZED GOSSIP
'''
def random_gossip_noW(graph, TOL):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        uniformly select a neighbor of node i
        Two methods to average:
            ''manual averaging of i and neighbor i values'' -- USED HERE
                x(k) = avg(x_i, one_rand_neighbor)
                e(k) = || . ||
            ''using the W matrix'' 
                construct W matrix
                check that W matrix obeys rules
                x(k) = W(k)x(k-1)
                e(k) = || . ||
                x(k-1) = x(k)
    TRANSMISSIONS: per iteration of while loop = 1
    * decided to only implement the without W implementation because matrix multiplications implodes the time
    '''
    print("")
    print("------- RANDOM GOSSIP (no W)------- ")

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
        
        # uniformly select a neighbor of node i      
        neighbors_i = list(nx.all_neighbors(graph, node_i))
        rand_neigh = random.choice(neighbors_i) 

        # x(k) = avg(x_i, one_rand_neighbor)
        cur_temp = all_temps[node_i]
        neigh_temp = all_temps[rand_neigh]
        avg = (cur_temp + neigh_temp)/2
        all_temps.update({node_i: avg, rand_neigh: avg})

        # TRANSMISSIONS: per iteration of while loop = 1
        transmissions += 1

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2))

    if np.max(np.mean(list(all_temps.values())) - true_avg) > TOL:
        print(f"\033[91mERROR: On average the values in x_k are not within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions

'''
PDMM SYNCHRONOUS
'''
def pdmm_synch(graph, TOL, c=0.3):
    '''
    1) Initialize variables
    x_0 = 0                                     (dimension = # nodes (n) x 1)
    a = sensor measurements vector              (dimension = # nodes (n) x 1)
    z_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    y_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    c = 0.1 (based on graph, good initial point)
    d = graph degree vector                     (dimension = # nodes (n) x 1)
    A = not adjacency matrix  (make method)     (dimension = # edges (m) x # nodes (n)) # implemented as dict

    2) while e(k) > epsilon:
        for all nodes i,
            update x_i(k) = ( a_i - sum(A_ij*z_ij(k-1)) ) / (1 + c*d_i)          # a_ij*z_ij(k-1) -> sum of all neighbors of i
            for all neighbors of i called j,
                update y_ij(k) = z_ij(k-1) + 2*c*x_i(k)*A_ij
        for all nodes i,
            for all N(i), 
                Send to node j the value y_ij. Node j will see it as y_ji.       # Due to implementation, this step can be skipped
                transmissions += 1
        for all nodes i,
            for all neighbors of i called j,
                z_ij = y_ji  
        e(k) = ||a - true_avg||_2^2
    TRANSMISSIONS: for all nodes i, for N(i), one transmission made
    '''
    print("")
    print("------- PDMM Synchronous ------- ")

    start_time = time.time()

    # Initialize variables
    all_nodes = list(nx.nodes(graph))
    all_edges = list(nx.edges(graph))
    a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
    x = np.zeros(len(all_nodes))
    list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]

    # Dimension of these should always be 2*num_edges
    z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    d = np.array([graph.degree(node) for node in all_nodes])

    # Make A: Implemented as a dictionary to avoid indexing issues
    A = {}
    for i, edge in enumerate(all_edges):
        A[(edge[0], edge[1])] = 1
        A[(edge[1], edge[0])] = -1
    
    # Get true average, used for stopping criterion
    true_avg = np.mean(a)
    std_devs = []
    errors = []
    transmissions = 0
    
    while (np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        for i in all_nodes:
            transmissions += 1
            x[i] = (a[i] - np.sum( A[(i, j)] * z_ij[(i, j)] for j in list_neighbors[i])) / (1 + c * d[i])
            for j in list_neighbors[i]:
                y_ij[(i, j)] = z_ij[(i, j)] + 2 * c * x[i] * A[(i, j)]
        for i in all_nodes:
            for j in list_neighbors[i]:
                z_ij[(i, j)] = y_ij[(j, i)]
        
        std_dev = statistics.stdev(x)
        std_devs.append((transmissions, std_dev))

        errors.append((transmissions, np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x[0])

    return x[0], std_devs, errors, transmissions

'''
PDMM ASYNCHRONOUS
'''
def pdmm_async(graph, TOL, c=0.4):
    '''
    1) Initialize variables
    x_0 = 0                                     (dimension = # nodes (n) x 1)
    a = sensor measurements vector              (dimension = # nodes (n) x 1)
    z_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    y_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    c = 0.1 (based on graph, good initial point)
    d = graph degree vector                     (dimension = # nodes (n) x 1)
    A = not adjacency matrix  (make method)     (dimension = # edges (m) x # nodes (n)) # implemented as dict

    2) while e(k) > epsilon:
        select a random node i
            update x_i(k) = ( a_i - sum(A_ij*z_ij(k-1)) ) / (1 + c*d_i)          # a_ij*z_ij(k-1) -> sum of all neighbors of i
            for all neighbors of i called j,
                update y_ij(k) = z_ij(k-1) + 2*c*x_i(k)*A_ij
        for all nodes i,
            for all N(i), 
                Send to node j the value y_ij. Node j will see it as y_ji.       # Due to implementation, this step can be skipped
                transmissions += 1
        for the single randomly selected node i,
            for all neighbors of i called j,
                z_ij = y_ji  
        e(k) = ||a - true_avg||_2^2
    TRANSMISSIONS: for all nodes i, for N(i), one transmission made
    UNICAST VERSION
    '''
    print("")
    print("------- PDMM Asynchronous ------- ")

    start_time = time.time()

    # Initialize variables
    all_nodes = list(nx.nodes(graph))
    all_edges = list(nx.edges(graph))
    a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
    x = np.zeros(len(all_nodes))
    list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]

    # Dimension of these should always be 2*num_edges
    z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    d = np.array([graph.degree(node) for node in all_nodes])

    # Make A: Implemented as a dictionary to avoid indexing issues
    A = {}
    for i, edge in enumerate(all_edges):
        A[(edge[0], edge[1])] = 1
        A[(edge[1], edge[0])] = -1
    
    # Get true average, used for stopping criterion
    true_avg = np.mean(a)
    std_devs = []
    errors = []
    transmissions = 0
    
    while (np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        i = random.choice(all_nodes)
        transmissions += 1
        x[i] = (a[i] - np.sum( A[(i, j)] * z_ij[(i, j)] for j in list_neighbors[i])) / (1 + c * d[i])
        for j in list_neighbors[i]:
            y_ij[(i, j)] = z_ij[(i, j)] + 2 * c * x[i] * A[(i, j)]
        for j in list_neighbors[i]:
            z_ij[(i, j)] = y_ij[(j, i)]
        
        std_dev = statistics.stdev(x)
        std_devs.append((transmissions, std_dev))

        errors.append((transmissions, np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x[0])

    return x[0], std_devs, errors, transmissions

'''
RANDOMIZED GOSSIP WITH TRANSMISSION FAILURES
'''
def random_gossip_TF(graph, TOL, FAILURE_RATE=0.0):
    '''
    1) x(k-1) = x(0)
    2) while e(k) > epsilon:
        uniformly select random node i
        uniformly select a neighbor of node i
        Two methods to average:
            ''manual averaging of i and neighbor i values'' -- USED HERE
                x(k) = avg(x_i, one_rand_neighbor)
                e(k) = || . ||
            ''using the W matrix'' 
                construct W matrix
                check that W matrix obeys rules
                x(k) = W(k)x(k-1)
                e(k) = || . ||
                x(k-1) = x(k)
    TRANSMISSIONS: per iteration of while loop = 1
    * decided to only implement the without W implementation because matrix multiplications implodes the time
    '''
    print("")
    print("------- RANDOM GOSSIP w/ Transmission Failures------- ")

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
        if (random.random() < FAILURE_RATE):        # 25% of the time, transmission fails, skip loop iteration
            transmissions += 1
            continue

        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # uniformly select a neighbor of node i      
        neighbors_i = list(nx.all_neighbors(graph, node_i))
        rand_neigh = random.choice(neighbors_i) 

        # x(k) = avg(x_i, one_rand_neighbor)
        cur_temp = all_temps[node_i]
        neigh_temp = all_temps[rand_neigh]
        avg = (cur_temp + neigh_temp)/2
        all_temps.update({node_i: avg, rand_neigh: avg})

        # TRANSMISSIONS: per iteration of while loop = 1
        transmissions += 1

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2))

    if np.max(np.mean(list(all_temps.values())) - true_avg) > TOL:
        print(f"\033[91mERROR: On average the values in x_k are not within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions

'''
RANDOMIZED GOSSIP WITH NODE DROP/ADD
'''
def random_gossip_dropadd(graph, TOL, DROP_RATE=0.0, ADD_RATE=0.0, type="bulk"):
    '''
    Random Gossip Algorithm
    4 OPTIONS:
        * Dropping nodes in bulk
        * Droppping nodes sequentially
        * Adding nodes in bulk
        * Adding nodes sequentially
    Access these options by setting EITHER DROP_RATE > 0.0 OR ADD_RATE > 0.0
    and by setting type to either "bulk" or "seq"
    '''
    print("")
    if (DROP_RATE > 0.0): # Dropping nodes
        print("------- RANDOM GOSSIP w/ " + str(type) + " DROP: " + str(DROP_RATE) + " ------- ")
    elif (ADD_RATE > 0.0): # Adding nodes
        print("------- RANDOM GOSSIP w/ " + str(type) + " ADD: " + str(ADD_RATE) + " ------- ")
    else: print("------- RANDOM GOSSIP w/ DROP: " + str(DROP_RATE) + " ADD " + str(ADD_RATE) + " ------- ")

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
    num_nodes_dropped_already = 0

    num_nodes_add = int(len(all_nodes) * ADD_RATE) # in total this is how many nodes you want to add
    print("Number of nodes to add: ", num_nodes_add)
    num_nodes_added_already = 0

    
    while (np.linalg.norm(list(all_temps.values()) - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        
        # CASE 1: BULK DROP
        if (transmissions > 10000 and DROPPED_FLAG == False and type == "bulk" and DROP_RATE > 0.0):
            graph_old = graph.copy()
            print("Number of nodes BEFORE DROP: ", len(all_nodes))
            num_nodes_drop = int(len(all_nodes) * DROP_RATE)
            # print("Number of nodes to drop: ", num_nodes_drop)
            nodes_to_drop = random.sample(all_nodes, num_nodes_drop)
            for node in nodes_to_drop:
                # print("Node to drop: ", node)
                graph.remove_node(node)
            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been dropped in bulk, the graph is no longer connected.\033[0m")
                break

            plot_rgg_side_by_side(graph_old, graph, "Bulk Drop")
            
            # Recalculation after nodes drop
            all_nodes = list(nx.nodes(graph))
            print("Number of nodes AFTER BULK DROP: ", len(all_nodes))
            # all_temps = nx.get_node_attributes(graph, "temp")
            all_temps = {node: temp for node, temp in all_temps.items() if node in graph}

            true_avg = np.mean(list(all_temps.values()))
            
            DROPPED_FLAG = True

        # CASE 2: SEQUENTIAL DROP
        if (transmissions > 10000 and DROPPED_FLAG == False and type == "seq" and transmissions % 1000 == 0 and DROP_RATE > 0.0): # only drop every 100 iterations of the while loop
            all_nodes = list(nx.nodes(graph))           # get current list of nodes
            node_to_drop = random.choice(all_nodes)     # drop a single random node from the graph
            graph.remove_node(node_to_drop) # drop a single random node from the graph

            # recalculate necessary things for computations later to continue
            # all_temps = nx.get_node_attributes(graph, "temp")
            all_temps = {node: temp for node, temp in all_temps.items() if node in graph}
            all_nodes = list(nx.nodes(graph))
            true_avg = np.mean(list(all_temps.values()))

            num_nodes_dropped_already += 1

            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been dropped in sequence, the graph is no longer connected.\033[0m")
                break

            if num_nodes_dropped_already == num_nodes_drop:
                print("Number of nodes AFTER SEQ DROP: ", len(all_nodes))
                DROPPED_FLAG = True
        
        # CASE 3: BULK ADD
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
                RADIUS = np.sqrt(np.log(2*len(all_nodes)+i+1)) / len(all_nodes)+i+1
                # print("Radius: ", RADIUS)
                for node in graph.nodes():
                    if node != len(all_nodes) + i + 1:
                        distance = np.linalg.norm(np.array(graph.nodes[node]['pos']) - np.array(graph.nodes[len(all_nodes) + i + 1]['pos']))
                        if distance <= RADIUS:
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
            true_avg = np.mean(list(all_temps.values()))
            
            ADDED_FLAG = True
        
        # CASE 4: SEQUENTIAL ADD
        if (transmissions > 5000 and ADDED_FLAG == False and type == "seq" and transmissions % 1000 == 0 and ADD_RATE > 0.0): # only add every 100 iterations of the while loop
            graph_old = graph.copy()
            all_nodes = list(nx.nodes(graph))           # get current list of nodes
            new_measurement = generate_measurements(1)
            graph.add_node(len(all_nodes)+1, pos=(random.uniform(0, 1), random.uniform(0, 1)), temp=new_measurement[0])
            pos = nx.get_node_attributes(graph, 'pos') 
            # print(len(pos), "should be equal to ", len(all_nodes) + 1)

            # randomly connect it to the graph in a random geometric manner
            RADIUS = np.sqrt(np.log(2*len(all_nodes)+1)) / len(all_nodes)+1
            # print("Radius: ", RADIUS)
            for node in graph.nodes():
                if node != len(all_nodes) + 1:
                    distance = np.linalg.norm(np.array(graph.nodes[node]['pos']) - np.array(graph.nodes[len(all_nodes) + 1]['pos']))
                    if distance <= RADIUS:
                        graph.add_edge(node, len(all_nodes) + 1)

            # to check this was done correctly, print the graph before and after adding singular node
            # plot_rgg_side_by_side(graph_old, graph)

            # recalculate necessary things for computations later to continue
            # all_temps = nx.get_node_attributes(graph, "temp")
            all_nodes = list(nx.nodes(graph))
            # print("Number of nodes in all_nodes: ", len(all_nodes))
            all_temps.update({node: graph.nodes[node]['temp'] for node in all_nodes[-num_nodes_add:]})
            # print("Number of nodes in all_temps: ", len(all_temps))
            true_avg = np.mean(list(all_temps.values()))

            num_nodes_added_already += 1

            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been dropped in sequence, the graph is no longer connected.\033[0m")
                break

            if num_nodes_added_already == num_nodes_add:
                print("Number of nodes AFTER SEQ DROP: ", len(all_nodes))
                ADDED_FLAG = True


        # uniformly select random node i
        node_i = random.choice(all_nodes)
        
        # uniformly select a neighbor of node i      
        neighbors_i = list(nx.all_neighbors(graph, node_i))
        rand_neigh = random.choice(neighbors_i) 

        # x(k) = avg(x_i, one_rand_neighbor)
        cur_temp = all_temps[node_i]
        neigh_temp = all_temps[rand_neigh]
        avg = (cur_temp + neigh_temp)/2
        all_temps.update({node_i: avg, rand_neigh: avg})

        # TRANSMISSIONS: per iteration of while loop = 1
        transmissions += 1

        # update values for plotting later
        std_dev = statistics.stdev(list(all_temps.values()))
        std_devs.append((transmissions, std_dev))

        # print("length of list of all_temps values", len(list(all_temps.values())))
        # e(k) = || . ||
        errors.append((transmissions, np.linalg.norm(list(all_temps.values()) - np.ones(len(list(all_temps.values()))) * true_avg)**2))

    if np.max(np.mean(list(all_temps.values())) - true_avg) > TOL:
        print(f"\033[91mERROR: On average the values in x_k are not within {TOL} of each other.\033[0m")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", list(all_temps.values())[0])

    return list(all_temps.values())[0], std_devs, errors, transmissions

'''
PDMM Asynchronous with Transmission Failures
'''
def pdmm_async_tf(graph, TOL, c=0.4, FAILURE_RATE=0.0):
    '''
    1) Initialize variables
    x_0 = 0                                     (dimension = # nodes (n) x 1)
    a = sensor measurements vector              (dimension = # nodes (n) x 1)
    z_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    y_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    c = 0.1 (based on graph, good initial point)
    d = graph degree vector                     (dimension = # nodes (n) x 1)
    A = not adjacency matrix  (make method)     (dimension = # edges (m) x # nodes (n)) # implemented as dict

    2) while e(k) > epsilon:
        if transmission failure case, skip iteration of while loop
        select a random node i
            update x_i(k) = ( a_i - sum(A_ij*z_ij(k-1)) ) / (1 + c*d_i)          # a_ij*z_ij(k-1) -> sum of all neighbors of i
            for all neighbors of i called j,
                update y_ij(k) = z_ij(k-1) + 2*c*x_i(k)*A_ij
        for all nodes i,
            for all N(i), 
                Send to node j the value y_ij. Node j will see it as y_ji.       # Due to implementation, this step can be skipped
                transmissions += 1
        for the single randomly selected node i,
            for all neighbors of i called j,
                z_ij = y_ji  
        e(k) = ||a - true_avg||_2^2
    TRANSMISSIONS: for all nodes i, for N(i), one transmission made
    UNICAST VERSION
    '''
    print("")
    print("------- PDMM Asynchronous TF ------- ")

    start_time = time.time()

    # Initialize variables
    all_nodes = list(nx.nodes(graph))
    all_edges = list(nx.edges(graph))
    a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
    x = np.zeros(len(all_nodes))
    list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]

    # Dimension of these should always be 2*num_edges
    z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    d = np.array([graph.degree(node) for node in all_nodes])

    # Make A: Implemented as a dictionary to avoid indexing issues
    A = {}
    for i, edge in enumerate(all_edges):
        A[(edge[0], edge[1])] = 1
        A[(edge[1], edge[0])] = -1
    
    # Get true average, used for stopping criterion
    true_avg = np.mean(a)
    std_devs = []
    errors = []
    transmissions = 0
    
    while (np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        if (random.random() < FAILURE_RATE):        # 25% of the time, transmission fails, skip loop iteration
            transmissions += 1
            continue

        i = random.choice(all_nodes)
        transmissions += 1
        x[i] = (a[i] - np.sum( A[(i, j)] * z_ij[(i, j)] for j in list_neighbors[i])) / (1 + c * d[i])
        for j in list_neighbors[i]:
            y_ij[(i, j)] = z_ij[(i, j)] + 2 * c * x[i] * A[(i, j)]
        for j in list_neighbors[i]:
            z_ij[(i, j)] = y_ij[(j, i)]
        
        std_dev = statistics.stdev(x)
        std_devs.append((transmissions, std_dev))

        errors.append((transmissions, np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x[0])

    return x[0], std_devs, errors, transmissions

'''
PDMM Asynchronous with Bulk Drop/Add
'''
def pdmm_asynch_dropadd(graph, TOL, c=0.4, DROP_RATE=0.0, ADD_RATE=0.0):
    '''
    1) Initialize variables
    x_0 = 0                                     (dimension = # nodes (n) x 1)
    a = sensor measurements vector              (dimension = # nodes (n) x 1)
    z_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    y_00 = 0                                    (dimension = [[# N(1) x 1 ], [# N(2) x 1], ... , [# N(n) x 1]]) # implemented as dict
    c = 0.1 (based on graph, good initial point)
    d = graph degree vector                     (dimension = # nodes (n) x 1)
    A = not adjacency matrix  (make method)     (dimension = # edges (m) x # nodes (n)) # implemented as dict

    2) while e(k) > epsilon:
        If iteration is 2000:
            Calculate how many nodes to drop
            Drop the nodes starting from the highest number (as not to mess with indexing)
            Recalculate all necessary variables
        select a random node i
            update x_i(k) = ( a_i - sum(A_ij*z_ij(k-1)) ) / (1 + c*d_i)          # a_ij*z_ij(k-1) -> sum of all neighbors of i
            for all neighbors of i called j,
                update y_ij(k) = z_ij(k-1) + 2*c*x_i(k)*A_ij
        for all nodes i,
            for all N(i), 
                Send to node j the value y_ij. Node j will see it as y_ji.       # Due to implementation, this step can be skipped
                transmissions += 1
        for the single randomly selected node i,
            for all neighbors of i called j,
                z_ij = y_ji  
        e(k) = ||a - true_avg||_2^2
    TRANSMISSIONS: for all nodes i, for N(i), one transmission made
    UNICAST VERSION
    '''
    print("")
    print("------- PDMM Asynchronous Bulk Drop/Add ------- ")

    start_time = time.time()

    # Initialize variables
    all_nodes = list(nx.nodes(graph))
    all_edges = list(nx.edges(graph))
    a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
    # Get true average, used for stopping criterion
    true_avg = np.mean(a)
    x = np.zeros(len(all_nodes))
    list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]

    # Dimension of these should always be 2*num_edges
    z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}      # check every time you update that its a valid edge
    d = np.array([graph.degree(node) for node in all_nodes])

    # Make A: Implemented as a dictionary to avoid indexing issues
    A = {}
    for i, edge in enumerate(all_edges):
        A[(edge[0], edge[1])] = 1
        A[(edge[1], edge[0])] = -1
    
    std_devs = []
    errors = []
    transmissions = 0
    DROPPED_FLAG = False
    ADDED_FLAG = False
    num_nodes_drop = 0

    while (np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2 > TOL):
        # if (transmissions % 100 == 0):
        #     print(np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2)
        
        # BULK DROP
        if (transmissions > 2000 and DROP_RATE > 0.0 and DROPPED_FLAG == False):
            graph_old = graph.copy()
            print("Number of nodes BEFORE DROP: ", len(all_nodes))
            num_nodes_drop = int(len(all_nodes) * DROP_RATE)
            nodes_to_drop = all_nodes[-num_nodes_drop:]
            print(nodes_to_drop)
            graph.remove_nodes_from(nodes_to_drop)

            # Check visually that nodes were removed
            plot_rgg_side_by_side(graph_old, graph, "Bulk Drop")

            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been dropped in bulk, the graph is no longer connected.\033[0m")
                break

            # Recalculate necessary variables
            all_nodes = list(nx.nodes(graph))
            print("Number of nodes AFTER DROP: ", len(all_nodes))
            all_edges = list(nx.edges(graph))
            a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
            # x = np.zeros(len(all_nodes))
            x = x[:-num_nodes_drop]
            # print("size of x is ", len(x))
            list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]
            true_avg = np.mean(a)

            z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}
            y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}
            d = np.array([graph.degree(node) for node in all_nodes])

            A = {}
            for i, edge in enumerate(all_edges):
                if edge[0] in all_nodes and edge[1] in all_nodes:
                    A[(edge[0], edge[1])] = 1
                    A[(edge[1], edge[0])] = -1
            DROPPED_FLAG = True
            print(DROPPED_FLAG)

        # BULK ADD
        if (transmissions > 2000 and ADD_RATE > 0.0 and ADDED_FLAG == False):
            old_num_nodes = len(all_nodes)
            graph_old = graph.copy()
            print("Number of nodes BEFORE ADD: ", len(all_nodes))
            num_nodes_add = int(len(all_nodes) * ADD_RATE)
            new_measurements = generate_measurements(num_nodes_add)

            # Add nodes to graph and connect them in a random geometric manner
            for i in range(num_nodes_add):
                # graph.add_node(len(all_nodes)+1+i, pos=(random.uniform(0, 1), random.uniform(0, 1)), temp=new_measurements[i])
                graph.add_node(len(all_nodes)+i, pos=(random.uniform(0, 1), random.uniform(0, 1)), temp=new_measurements[i])
                pos = nx.get_node_attributes(graph, 'pos') 
                # RADIUS = np.sqrt(np.log(2*len(all_nodes)+i+1)) / len(all_nodes)+i+1
                # RADIUS = np.sqrt(np.log(2*len(all_nodes)+i)) / len(all_nodes)+i

                RADIUS = np.sqrt(np.log(2*old_num_nodes) / old_num_nodes)

                for node in graph.nodes():
                    # if node != len(all_nodes) + i + 1:
                    if node != len(all_nodes) + i:
                        # distance = np.linalg.norm(np.array(graph.nodes[node]['pos']) - np.array(graph.nodes[len(all_nodes) + i + 1]['pos']))
                        distance = np.linalg.norm(np.array(graph.nodes[node]['pos']) - np.array(graph.nodes[len(all_nodes) + i]['pos']))
                        if distance <= RADIUS:
                            # graph.add_edge(node, len(all_nodes) + i + 1)
                            graph.add_edge(node, len(all_nodes) + i)
            
            # Check visually that nodes were added
            plot_rgg_side_by_side(graph_old, graph, "Bulk Add")

            # Check if the graph is connected
            if not nx.is_connected(graph):
                print("\033[91mERROR: After the nodes have been added in bulk, the graph is no longer connected.\033[0m")
                break

            # Recalculate necessary variables
            all_nodes = list(nx.nodes(graph))
            print("Number of nodes AFTER ADD: ", len(all_nodes))
            all_edges = list(nx.edges(graph))
            a = np.array(list(nx.get_node_attributes(graph, "temp").values()))
            # x = np.zeros(len(all_nodes))
            x = np.concatenate((x, np.zeros(num_nodes_add)))
            # print("size of x is ", len(x))
            list_neighbors = [list(nx.all_neighbors(graph, node)) for node in all_nodes]
            true_avg = np.mean(a)

            z_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}
            y_ij = {(i, j): 0.0 for i in all_nodes for j in list_neighbors[i]}
            d = np.array([graph.degree(node) for node in all_nodes])

            A = {}
            for i, edge in enumerate(all_edges):
                if edge[0] in all_nodes and edge[1] in all_nodes:
                    A[(edge[0], edge[1])] = 1
                    A[(edge[1], edge[0])] = -1
            
            ADDED_FLAG = True
            print(ADDED_FLAG)

        all_nodes = list(nx.nodes(graph))
        # print(all_nodes)
        i = random.choice(all_nodes)
        transmissions += 1
        
        x[i] = (a[i] - np.sum( A[(i, j)] * z_ij[(i, j)] for j in list_neighbors[i])) / (1 + c * d[i])
        for j in list_neighbors[i]:
            y_ij[(i, j)] = z_ij[(i, j)] + 2 * c * x[i] * A[(i, j)]
        for j in list_neighbors[i]:
            z_ij[(i, j)] = y_ij[(j, i)]
        
        std_dev = statistics.stdev(x)
        std_devs.append((transmissions, std_dev))
        errors.append((transmissions, np.linalg.norm(x - np.ones(len(all_nodes)) * true_avg)**2))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", x[0])

    return x[0], std_devs, errors, transmissions
    
