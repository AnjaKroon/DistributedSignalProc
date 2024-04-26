import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def build_random_graph(num_nodes_fix,required_probability=0.999,fix_num_nodes=False):
    """
    Builds a random geometric graph with a given required probability.

    Parameters:
    - required_probability (float): The required probability for the graph connectivity. Default is 0.999.

    Returns:
    - num_nodes (int): The number of nodes in the graph.
    - G (networkx.Graph): The generated random geometric graph.
    - A (numpy.ndarray): The adjacency matrix of the graph.
    - pos (dict): The positions of the nodes in the graph.
    """
    # we are working in 2 dimensions
    num_nodes = int(np.ceil(np.sqrt(1 / (1 - required_probability))))
    if fix_num_nodes:
        num_nodes=num_nodes_fix
    
    r_c = np.sqrt(np.log(2*num_nodes) / num_nodes)

    pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

    G = nx.random_geometric_graph(n=num_nodes, radius=r_c * 100, pos=pos)

    A = nx.adjacency_matrix(G).toarray()
    return num_nodes, G, A, pos

def add_node_to_graph(G, pos, new_node_id):
    """
    Add a new node to an existing graph, connecting it to all nodes within a certain radius.

    Args:
    - G (nx.Graph): The existing graph.
    - pos (dict): The positions of the nodes in the graph.
    - new_node_id (int): The id of the new node.

    Returns:
    - G (nx.Graph): The updated graph.
    - pos (dict): The updated positions of the nodes in the graph.
    - A (np.array): The adjacency matrix of the updated graph.
    """
    num_nodes = len(G.nodes)
    r_c = np.sqrt(np.log(2*num_nodes) / num_nodes)

    # Generate a random position for the new node
    new_pos = (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100))

    # Add the new node to the graph and the position dictionary
    G.add_node(new_node_id)
    pos[new_node_id] = new_pos

    # Connect the new node to all nodes within the radius
    for node, node_pos in pos.items():
        if np.linalg.norm(np.array(node_pos) - np.array(new_pos)) <= r_c * 100:
            if new_node_id !=node: 
                G.add_edge(new_node_id, node)

    return G, pos

def W_construct_rand_gossip(i, j, n):
    """
    Constructs a weight matrix W for a distributed signal processing system.

    Parameters:
    i (int): Index of the first element.
    j (int): Index of the second element.
    n (int): Number of elements in the system.

    Returns:
    numpy.ndarray: Weight matrix W.

    """
    e_i = np.zeros([n, 1])
    e_j = np.zeros([n, 1])
    e_i[i] = 1
    e_j[j] = 1
    W = np.identity(n)
    W -= 0.5 * (e_i - e_j) * np.transpose(e_i - e_j)
    return W

def generate_temp_field(num_nodes, var, true_temp):
    """
    Generate a temperature field for a given number of nodes.

    Parameters:
    - num_nodes (int): The number of nodes in the field.
    - var (float): The variance of the temperature values.
    - true_temp (float): The true temperature value.

    Returns:
    - temperature (numpy.ndarray): An array of shape (num_nodes, 1) containing the generated temperature values.
    """
    temperature = np.zeros([num_nodes, 1])

    for i, val in enumerate(temperature):
        temperature[i][0] += np.random.normal(true_temp, np.sqrt(var))

    return temperature

def plot_log_convergence(losses,transmissions, legend,num_nodes):
    for i,loss in enumerate(losses):
        plt.plot(transmissions[i], loss)
    plt.xlabel('Transmissions')
    plt.ylabel('Loss ||x-x_ave||^2')
    plt.title('Loss vs Transmission for {} nodes'.format(num_nodes))
    plt.yscale('log')
    plt.legend(legend)
    plt.show()

 