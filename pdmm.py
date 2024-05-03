
import networkx as nx
import numpy as np
import random
import statistics
import time

from utils import generate_measurements, generate_rgg, vector_to_dict
from visualization import plot_rgg_side_by_side


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
    
