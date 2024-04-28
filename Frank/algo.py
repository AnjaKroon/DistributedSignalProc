import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
# np.random.seed(10)

####Asynchronous algorithms
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

def random_gossip_TF(temperature, A, tolerance=0.00001, transmission_failure=0.1):
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
        #transmission failure
        if np.random.uniform(low=0, high=1)<transmission_failure:
            loss= np.append(loss, np.sum((temperature - avg_temp)**2))
            continue
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

def random_gossip_node_change(temperature, G,pos,true_temp,var, tolerance=0.00001,node_change_status="add_bulk",averaging_method="update", max_iter=100000):
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
    A = nx.adjacency_matrix(G).toarray()
    num_nodes = np.shape(A)[0]
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)
    iter=0
    while not converged and iter<max_iter:
        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.transpose(np.nonzero(A[node_i, :]))
        j_index = int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
        node_j = i_neigh[j_index][0]

        # update equation
        # W = W_construct_rand_gossip(node_i, node_j, num_nodes)
        # temperature = np.dot(W, temperature)
        avg=(temperature[node_i]+temperature[node_j])/2
        temperature[node_i]=avg
        temperature[node_j]=avg

        #implement removing/adding of nodes
        if node_change_status=="add_bulk" and  iter ==5000:
            for i in range(1,40):
                #add one node, update graph, temperature and number of nodes
                num_nodes =num_nodes+1
                G,pos=add_node_to_graph(G,pos,num_nodes)
                A = nx.adjacency_matrix(G).toarray()
                new_temp=np.random.normal(true_temp, np.sqrt(var))
                temperature=np.append(temperature,new_temp)
                if averaging_method=="update":
                    avg_temp = (avg_temp*(num_nodes-1)+new_temp)/num_nodes
        elif node_change_status=="remove_bulk" and  iter ==5000:
            for i in range(1,40):
                #remove one node, update graph, temperature and number of nodes
                node_ids_list = sorted(list(G.nodes()))
                G.remove_node(node_ids_list[-1])
                num_nodes =num_nodes-1
                A = nx.adjacency_matrix(G).toarray()
                old_temp=temperature[num_nodes]
                temperature=np.delete(temperature,-1)
                if averaging_method=="update":
                    avg_temp = (avg_temp*(num_nodes+1)-old_temp)/num_nodes

        iter =iter+1
        if np.sum((temperature - avg_temp)**2)< tolerance:
            loss = np.append(loss, np.sum((temperature - avg_temp)**2))
            converged = True
        else:
            # print(iter,"\t",np.sum((temperature - avg_temp)**2))
            loss= np.append(loss, np.sum((temperature - avg_temp)**2))
    return loss,temperature

def PDMM_async(temperature, G,tolerance=10**-8,c=0.3):
    num_nodes = G.number_of_nodes()
    x=np.zeros([num_nodes,1])
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)
    #initialise A_ij
    A_ij=calc_incidence(G)

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
        i = int(np.random.uniform(low=0, high=num_nodes))
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
            print(np.sum((x- avg_temp)**2))
            loss= np.append(loss, np.sum((x - avg_temp)**2))
  
        #update z_ij
        for j in G.neighbors(i):
            z[i,j]=y[j,i]
                
    return loss,transmissions

def PDMM_async_TF(temperature, G, tolerance=10**-8, c=0.3, transmission_failure=0.1):
    num_nodes = G.number_of_nodes()
    x = np.zeros([num_nodes, 1])
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)

    # Initialise A_ij
    A_ij=calc_incidence(G)
    # Initialise z_ij and y_ij
    z = dict()
    y = dict()
    for i in np.arange(0, num_nodes):
        for j in G.neighbors(i):
            z[i, j] = 0
            y[i, j] = 0

    transmissions = []
    tot_transmissions = 0
    while not converged:
        i = int(np.random.uniform(low=0, high=num_nodes))
        if np.random.uniform(low=0, high=1) < transmission_failure:
            tot_transmissions += 1
            transmissions.append(tot_transmissions)
            loss = np.append(loss, np.sum((x - avg_temp)**2))
            continue

        # Update x_i
        x[i] = temperature[i]
        for j in G.neighbors(i):
            x[i] -= A_ij[i, j] * z[i, j]
        x[i] /= (1 + c * G.degree(i))

        # Update y_ij
        for j in G.neighbors(i):
            y[i, j] = z[i, j] + 2 * c * (x[i] * A_ij[i, j])

        tot_transmissions += 1
        transmissions.append(tot_transmissions)
        loss = np.append(loss, np.sum((x - avg_temp)**2))

        if np.sum((x - avg_temp)**2) < tolerance:
            converged = True

        # Update z_ij
        for j in G.neighbors(i):
            z[i, j] = y[j, i]

    return loss, transmissions

def PDMM_async_node_change(temperature, G,pos,true_temp,var,tolerance=10**-8,c=0.3,node_change_status="add_bulk",averaging_method="update", max_iter=100000):
    num_nodes = G.number_of_nodes()
    x=np.zeros([num_nodes,1])
    converged = False
    loss = np.array([])
    avg_temp = np.mean(temperature)
    #initialise A_ij
    A_ij=calc_incidence(G)

    #initialise z_ij and y_ij
    z=dict()
    y=dict()
    for i in np.arange(0,num_nodes):
        for j in G.neighbors(i):
            z[i,j]=0
            y[i,j]=0
            
    transmissions=[]
    tot_transmissions=0
    iter=0
    while not converged and iter<max_iter:
        iter = iter+1
        
        #update x_i and y_ij
        i = int(np.random.uniform(low=0, high=num_nodes))
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
            print(np.sum((x- avg_temp)**2))
            loss= np.append(loss, np.sum((x - avg_temp)**2))

        #update z_ij
        for j in G.neighbors(i):
            z[i,j]=y[j,i]
        
        #implement removing/adding of nodes
        if node_change_status=="add_bulk" and  iter ==5000:
            for i in range(1,100):
                #add one node, update graph, temperature and number of nodes
                G,pos=add_node_to_graph(G,pos,num_nodes)
                num_nodes =num_nodes+1
                new_temp=np.random.normal(true_temp, np.sqrt(var))
                temperature=np.append(temperature,new_temp)
                x=np.append(x,0)
                #initialise dictionary values
                for j in G.neighbors(num_nodes-1):
                    z[num_nodes-1,j]=0
                    z[j,num_nodes-1]=0
                    y[num_nodes-1,j]=0
                    y[j,num_nodes-1]=0
                A_ij=calc_incidence(G)
                if averaging_method=="update":
                    avg_temp = (avg_temp*(num_nodes-1)+new_temp)/num_nodes
        elif node_change_status=="remove_bulk" and  iter ==5000:
            for i in range(1,20):
                #remove one node, update graph, temperature and number of nodes
                node_ids_list = sorted(list(G.nodes()))
                G.remove_node(node_ids_list[-1])
                num_nodes =num_nodes-1
                old_temp=temperature[num_nodes]
                temperature=np.delete(temperature,-1)
                x=np.delete(x,-1)
                
                if averaging_method=="update":
                    avg_temp = (avg_temp*(num_nodes+1)-old_temp)/num_nodes

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

####Synchronous algorithm
def PDMM_sync(temperature, G,tolerance=10**-8,c=0.1):
    num_nodes = G.number_of_nodes()
    x=np.zeros([num_nodes,1])
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

