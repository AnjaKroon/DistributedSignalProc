
import networkx as nx
import numpy as np
import random
import statistics
import time

'''
SYNCH. DIST AVG
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
def dist_avg_synch(graph, TOL):
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
def dist_avg_asynch_W(graph, TOL):
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
def dist_avg_asynch_noW(graph, TOL):
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
RANDOMIZED GOSSIP
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
def random_gossip_noW(graph, TOL):
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
PDMM Synchronous
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
def pdmm_synch(graph, TOL, c=0.3):
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
PDMM Asynchronous
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
def pdmm_async(graph, TOL, c=0.4):
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
RANDOMIZED GOSSIP TRANSMISSION FAILURES
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
def random_gossip_TF(graph, TOL, FAILURE_RATE=0.25):
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
PDMM Asynchronous with Transmission Failures
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
def pdmm_async(graph, TOL, c=0.4):
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
