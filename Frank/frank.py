import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
from algo import *


# np.random.seed(1)



required_probability=0.9999
num_nodes, G,A,pos=build_random_graph(50,required_probability,fix_num_nodes=True)
print("num_nodes:",num_nodes)

#now to generate measured values for the temperature sensors ins some flat 3d field
temperature=generate_temp_field(num_nodes,10,25)
tolerance=10**-12


loss_random,temperature_rand=random_gossip(temperature.copy(),A,tolerance)
loss_random_add,temperature_rand_add=random_gossip_node_change(temperature.copy(),G,pos,25,10,tolerance,"add_bulk")
plot_log_convergence([loss_random_add, loss_random],[np.arange(1,loss_random_add.shape[0]+1),np.arange(1,loss_random.shape[0]+1)],['Random Gossip add',"Random gossip"],num_nodes)






#sweep the c parameter
# trans=[]
# for c in np.arange(0.05,0.82,0.05):
#     print(c)
#     loss_pdmm_sync,trans_pdmm_sync=PDMM_sync(temperature,G,tolerance,c)    
#     trans.append(trans_pdmm_sync[-1])


# plt.plot(np.arange(0.05,0.82,0.05),trans)
# plt.xlabel('c')
# plt.ylabel('Transmissions')
# plt.show()
c=0.3

# loss_random,temperature_rand=random_gossip(temperature.copy(),A,tolerance)
# loss_random_tf,temperature_rand_tf=random_gossip_TF(temperature.copy(),A,tolerance,0.5)
# plot_log_convergence([loss_random_tf, loss_random],[np.arange(1,loss_random_tf.shape[0]+1),np.arange(1,loss_random.shape[0]+1)],['Random Gossip TF',"Random gossip"],num_nodes)

# loss_pdmm_async_tf,trans_pdmm_async_tf=PDMM_async_TF(temperature,G,tolerance,c,0.5)
# loss_pdmm_async,trans_pdmm_async=PDMM_async(temperature,G,tolerance,c)
# plot_log_convergence([loss_pdmm_async,loss_pdmm_async_tf],[trans_pdmm_async,trans_pdmm_async_tf],['PDMM_Async ',"PDMM_Async_TF"],num_nodes)


# loss_pdmm_sync,trans_pdmm_sync=PDMM_sync(temperature.copy(),G,tolerance,c)
# loss_pdmm_async,trans_pdmm_async=PDMM_async(temperature.copy(),G,tolerance,c)
# loss_random,temperature_rand=random_gossip(temperature.copy(),A,tolerance)

# loss_async,trans_async, temperature_async =async_distr_averaging(temperature.copy(),A,tolerance)


# plot_log_convergence([loss_async,loss_random,loss_pdmm_sync,loss_pdmm_async],[trans_async,np.arange(1,loss_random.shape[0]+1),trans_pdmm_sync,trans_pdmm_async],['Asynchronous Distributed Averaging','Random Gossip','Synchronous PDMM',"Asynchronous PDMM"],num_nodes)


# #select a node i at random (uniformly) and contact neigbouring node j at random(uniformly)
# nx.draw(G, pos=pos)
# plt.show()

