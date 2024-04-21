import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def W_construct(i,j,n):
    e_i=np.zeros([n,1])
    e_j=np.zeros([n,1])
    e_i[i]=1;e_j[j]=1
    W=np.identity(n)
    W-=0.5*(e_i-e_j)*np.transpose(e_i-e_j)
    return W
# working_dim=[100,100]
# area=working_dim[0]*working_dim[1]

#assuming this is some kind of  adhoc network

#we want to construct a random geometric graph 

#r^d>2log(n)/n then probability G^d(n,r) is connected with with probability of atleast 1-1/n^2

# we are working in 2 dimensions (maybe 3)



# required_probability=0.9999
# num_nodes=int(np.ceil(np.sqrt(1/(1-required_probability))))

num_nodes=200
r_c=np.sqrt(np.log(num_nodes)/num_nodes)

print("Number of nodes: ",(num_nodes),"\t\tCritical radius for unity area: ", "{:2f}".format(r_c))

pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

G = nx.random_geometric_graph(n=num_nodes, radius=r_c*100, pos=pos)

A=nx.adjacency_matrix(G).toarray()


###################################################################################################
#now to generate measured values for the temperature sensors ins some flat 3d field
true_temp=25
sensor_var=5
temperature=np.zeros([num_nodes,1])

for i,val in enumerate(temperature):
    temperature[i][0]+=np.random.normal(true_temp,np.sqrt(sensor_var))

#select a node i at random (uniformly) and contact neigbouring node j at random(uniformly)
# lets first do the natural averaging


nx.draw(G, pos=pos, with_labels=True)
plt.show()


converged=False


num_iter=0


x_av=np.ones([num_nodes,1])*np.average(temperature)
temp_og=temperature
while(not converged):
# for var in range(100000):
    num_iter+=1
    node_i=int(np.random.uniform(low=0, high=num_nodes))
    i_neigh=np.transpose(np.nonzero(A[node_i,:]))
    j_index=int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))

    node_j=i_neigh[j_index][0]

    #update equation
    W=W_construct(node_i,node_j,num_nodes)
    
    temp_old=temperature
    #print("B",node_i,node_j,temperature[node_i],temperature[node_j],"neigh,index:",j_index)
    temperature=np.dot(W,temperature)
    #print("A",node_i,node_j,temperature[node_i],temperature[node_j])
    
    #print(sum(np.abs(x_av-temperature)),node_i,node_j)
    if sum(np.abs(x_av-temperature)) < 0.00001:
        converged=True
    elif num_iter%1000==0:
        print(num_iter)
    


print("new",np.transpose(temperature),"\nold",np.transpose(temp_old),"\n numiter:",num_iter)


