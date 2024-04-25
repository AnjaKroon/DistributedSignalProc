import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(1)


def build_random_graph(required_probability=0.999):
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

    num_nodes = 100
    r_c = np.sqrt(np.log(2*num_nodes) / num_nodes)

    pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

    G = nx.random_geometric_graph(n=num_nodes, radius=r_c * 100, pos=pos)

    A = nx.adjacency_matrix(G).toarray()
    return num_nodes, G, A, pos

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


def random_gossip(temperature, A, tolerance=0.00001):
    """
    Perform random gossip algorithm to update the temperature values.

    Parameters:
    temperature (numpy.ndarray): The initial temperature values for each node.
    num_nodes (int): The total number of nodes in the network.
    A (numpy.ndarray): The adjacency matrix representing the network connections.
    tolerance (float, optional): The convergence tolerance. Defaults to 0.00001.

    Returns:
    numpy.ndarray: The loss values at each iteration until convergence.

    """
    num_nodes = np.shape(A)[0]
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)
    while not converged:

        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.transpose(np.nonzero(A[node_i, :]))
        j_index = int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
        node_j = i_neigh[j_index][0]

        # update equation
        W = W_construct_rand_gossip(node_i, node_j, num_nodes)
        temperature = np.dot(W, temperature)
        # avg=(temperature[node_i]+temperature[node_j])/2
        # temperature[node_i]=avg
        # temperature[node_j]=avg

        if np.sum((temperature - avg_temp)**2)< tolerance:
            loss = np.append(loss, np.sum((temperature - avg_temp)**2))
            converged = True
        else:
            loss= np.append(loss, np.sum((temperature - avg_temp)**2))
    return loss,temperature


# def ADMM_async(temperature, A):
#     num_nodes = np.shape(A)[0]
#     converged = False
#     loss = np.array([])
#     avg_temp = np.mean(temperature)

#     p=0.1
#     z=np.zeros([num_nodes,num_nodes])
#     v=np.zeros([num_nodes,num_nodes])
#     x=np.zeros([num_nodes,1])
#     while not converged:

#         node_i = int(np.random.uniform(low=0, high=num_nodes))
#         i_neigh = np.transpose(np.nonzero(A[node_i, :]))
#         j_index = int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
#         node_j = i_neigh[j_index][0]

#         #first update x_i
#         x[node_i]=temperature[node_i]
#         x[node_i]=x[node_i]+p*z[node_i,node_j]-v[node_i,node_j]
#         x[node_i]=x[node_i]/(1+p*np.shape(i_neigh)[0])

#         x[node_j]=temperature[node_j]
#         x[node_j]=x[node_j]+p*z[node_j,node_i]-v[node_j,node_i]
#         x[node_j]=x[node_j]/(1+p*np.shape(i_neigh)[0])

#         #next update z_ij
#         z[node_i,node_j]=(x[node_i]+x[node_j]+(1/p)*(v[node_i,node_j]+v[node_j,node_i]))/2
        
#         if np.sum((temperature - avg_temp)**2)< tolerance:
#             loss = np.append(loss, np.sum((temperature - avg_temp)**2))
#             converged = True
#         else:
#             loss= np.append(loss, np.sum((temperature - avg_temp)**2))
#     return loss,temperature


def PDMM_async(temperature, G,tolerance=0.00001):
    num_nodes = G.number_of_nodes()
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)
    incidence_matrix = nx.incidence_matrix(G).toarray().T
    Adj_matrix = nx.adjacency_matrix(G).toarray()

    A_ij=np.zeros(np.shape(incidence_matrix))
 

    x=np.zeros([num_nodes,1])
    # lambda_ji=np.zeros([np.shape(incidence_matrix)[0],1])
    y_ij=np.zeros([np.shape(incidence_matrix)[0],1])
    z_ij=np.zeros([np.shape(incidence_matrix)[0],1])
    c=0.4

    for i, edge in enumerate(G.edges):
        if edge[0] < edge[1]:
            A_ij[i, edge[0]-1] = 1
            A_ij[i, edge[1]-1] = -1
        else:
            A_ij[i, edge[0]-1] = -1
            A_ij[i, edge[1]-1] = 1

   

    while not converged:

        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.array(G.neighbors(node_i))
        # j_index = int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
        # node_j = i_neigh[j_index]

        #update x_i
        x[node_i]=temperature[node_i]
        for j in i_neigh:
            x[node_i]=x[node_i]-np.dot(A_ij[:,j],z_ij)
        x[node_i]=x[node_i]/(1+c*G.degree(node_i))



        if np.sum((temperature - avg_temp)**2)< tolerance:
            loss = np.append(loss, np.sum((temperature - avg_temp)**2))
            converged = True
        else:
            loss= np.append(loss, np.sum((temperature - avg_temp)**2))

    return 0

def PDMM_sync(temperature, G,tolerance=10**-8):
    num_nodes = G.number_of_nodes()
    c=0.1
    x=np.zeros([num_nodes,1])
    incidence_matrix = nx.incidence_matrix(G).toarray().T
    A_ij=dict()


    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)

    #initialise A_ij
    for edge in G.edges:
        if edge[0] < edge[1]:
            A_ij[edge[0], edge[1]] = 1
            A_ij[edge[1], edge[0]] = -1
        else:
            A_ij[edge[0], edge[1]] = -1
            A_ij[edge[1], edge[0]] = -1

    #initialise z_ij and y_ij
    z=dict()
    y=dict()
    for i in np.arange(0,num_nodes):
        for j in G.neighbors(i):
            z[i,j]=0
            y[i,j]=0
            
    transmissions=[]
    tot_transmissions=0
    while not converged:
        #update x_i and y_ij
        
        for i in  np.arange(0,num_nodes):
            #update x_i
            x[i]=temperature[i]
            for j in G.neighbors(i):
                x[i]=x[i]-A_ij[i,j]*z[i,j]
            x[i]=x[i]/(1+c*G.degree(i))
            #update y_ij
            for j in G.neighbors(i):
                y[i,j]=z[i,j]+2*c*(x[i]*A_ij[i,j])
            tot_transmissions=tot_transmissions+1
            transmissions.append(tot_transmissions)

            if np.sum((x- avg_temp)**2)< tolerance:
                loss = np.append(loss, np.sum((x - avg_temp)**2))
                converged = True
            else:
                # print(np.sum((x- avg_temp)**2))
                loss= np.append(loss, np.sum((x - avg_temp)**2))
  
        #update z_ij
        for i in  np.arange(0,num_nodes):
            for j in G.neighbors(i):
                z[i,j]=y[j,i]
                
        
        

    
    return loss,transmissions

            

def async_distr_averaging(temperature,A,tolerance):
    """
    Perform asynchronous distributed averaging algorithm.

    Returns:
    numpy.ndarray: The loss values at each iteration until convergence.

    """
    num_nodes = np.shape(A)[0]
    converged = False
    loss_a = np.array([])
    transmissions = np.array([])
    avg_temp = np.mean(temperature)
    while not converged:
        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.transpose(np.nonzero(A[node_i, :]))
        num_neigh = np.shape(i_neigh)[0]
        
        # update equation
        avg_val= (np.sum(temperature[i_neigh])+temperature[node_i]) / (num_neigh+1)
        transmissions = np.append(transmissions, num_neigh+transmissions[-1] if transmissions.size > 0 else num_neigh)
        temperature[node_i] = avg_val
        temperature[i_neigh] = avg_val

        if np.sum((temperature - avg_temp)**2)< tolerance:
            loss_a = np.append(loss_a, np.sum((temperature - avg_temp)**2))
            converged = True
        else:
            loss_a = np.append(loss_a, np.sum((temperature - avg_temp)**2))

    return loss_a,transmissions,temperature

def plot_log_convergence(losses,transmissions, legend,num_nodes):
    for i,loss in enumerate(losses):
        plt.plot(transmissions[i], loss)
    plt.xlabel('Transmissions')
    plt.ylabel('Loss ||x-x_ave||^2')
    plt.title('Loss vs Transmission for {} nodes'.format(num_nodes))
    plt.yscale('log')
    plt.legend(legend)
    plt.show()
    
required_probability=0.9999
num_nodes, G,A,pos=build_random_graph(required_probability)
print("num_nodes:",num_nodes)


#now to generate measured values for the temperature sensors ins some flat 3d field
temperature=generate_temp_field(num_nodes,10,25)

# PDMM_async(temperature, G)
loss_pdmm_sync,trans_pdmm_sync=PDMM_sync(temperature,G)


tolerance=10**-8
loss_random,temperature_rand=random_gossip(temperature.copy(),A,tolerance)

loss_async,trans_async, temperature_async =async_distr_averaging(temperature.copy(),A,tolerance)

plot_log_convergence([loss_async,loss_random,loss_pdmm_sync],[trans_async,np.arange(1,loss_random.shape[0]+1),trans_pdmm_sync],['Asynchronous Distributed Averaging','Random Gossip','Synchronous PDMM'],num_nodes)


#select a node i at random (uniformly) and contact neigbouring node j at random(uniformly)
# nx.draw(G, pos=pos)
# plt.show()
