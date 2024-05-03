
import networkx as nx
import numpy as np
import random
import statistics
import time

from utils import generate_measurements, generate_rgg, vector_to_dict
from visualization import plot_rgg_side_by_side


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

    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", all_temps[0])

    return all_temps[0], std_devs, errors, transmissions

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
    

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Transmissions: ", transmissions)
    print("Average: ", list(all_temps.values())[0])

    return list(all_temps.values())[0], std_devs, errors, transmissions


