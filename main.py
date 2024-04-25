# AUTHOR: Anja Kroon

# Code for distributed SP project
# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

from utils import generate_rgg, generate_measurements, vector_to_dict
from algorithms import dist_avg_synch, dist_avg_asynch_W, dist_avg_asynch_noW, random_gossip_noW, pdmm_synch
from visualization import plot_single_error, plot_multiple_pairs
import numpy as np

# MANUAL
# NODES = 500
# RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01

#AUTO
PROB_CONN = 0.999
NODES = int(np.ceil(np.sqrt(1 / (1 - PROB_CONN))))
print("Number of nodes: ", NODES)
NODES = 500
RAD = np.sqrt(np.log(2*NODES) / NODES)
DIM = 2
TOL = 10**-12

def main():
    temps = generate_measurements(NODES)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(NODES, RAD, DIM, dict_temps)

    # SYNCH DIST AVG
    avg_dist_avg_synch, stdev_dist_avg_synch, errors_synch, trans_synch = dist_avg_synch(rand_geo_gr, TOL)

    # ASYNCH DIST AVG
    avg_dist_avg_asynch_W, stdev_dist_avg_asynch_W, errors_asynch_W, trans_asynch_W = dist_avg_asynch_W(rand_geo_gr, TOL)
    avg_asynch_noW, stdev_asynch_noW, errors_asynch_noW, trans_asynch_noW = dist_avg_asynch_noW(rand_geo_gr, TOL)

    # RANDOM GOSSIP
    avg_rand_goss, stdev_rand_goss_noW, error_rand_goss_noW, trans_rand_goss_noW = random_gossip_noW(rand_geo_gr, TOL)

    # PLOTTING
    '''
    plot_single_error(errors_synch, "Synch Dist Avg")
    plot_single_error(errors_asynch_W, "Asynch Dist Avg with W")
    plot_single_error(errors_asynch_noW, "Asynch Dist Avg no W")
    plot_single_error(error_rand_goss_noW, "Random Gossip no W")

    plot_multiple_pairs(((errors_synch, "Synch Dist Avg"),
                        (errors_asynch_W, "Asynch Dist Avg with W"),
                        (errors_asynch_noW, "Asynch Dist Avg no W"), 
                        (error_rand_goss_noW, "Random Gossip no W")))
    
    plot_multiple_pairs(((errors_asynch_noW, "Asynch Dist Avg no W"), 
                        (error_rand_goss_noW, "Random Gossip no W")))
    '''

    # PDMM
    avg_pdmm_synch, stdev_pdmm_synch, error_pdmm_synch, trans_pdmm_synch = pdmm_synch(rand_geo_gr, TOL)    
    # plot_single_error(error_pdmm_synch, "PDMM Synch")

    plot_multiple_pairs(((errors_asynch_noW, "Asynch Dist Avg no W"), 
                        (error_rand_goss_noW, "Random Gossip no W"),
                        (error_pdmm_synch, "PDMM Synch")))

if __name__ == "__main__":
    main()
