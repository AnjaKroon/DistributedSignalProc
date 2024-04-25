# AUTHOR: Anja Kroon

# Code for distributed SP project
# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

from utils import generate_rgg, generate_measurements, vector_to_dict
from algorithms import dist_avg_synch, dist_avg_asynch_W, dist_avg_asynch_noW, random_gossip_noW, pdmm_synch, pdmm_async
from visualization import plot_single_error, plot_multiple_pairs, plot_c_transmissions
import numpy as np

# MANUAL
# NODES = 100
# RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01

NODES = 200
RAD = np.sqrt(np.log(2*NODES) / NODES)

#AUTO
# PROB_CONN = 0.9
# NODES = int(np.ceil(np.sqrt(1 / (1 - PROB_CONN))))

DIM = 2
TOL = 10**-12

def main():
    temps = generate_measurements(NODES)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(NODES, RAD, DIM, dict_temps)

    # SYNCH DIST AVG
    # avg_dist_avg_synch, stdev_dist_avg_synch, errors_synch, trans_synch = dist_avg_synch(rand_geo_gr, TOL)

    # ASYNCH DIST AVG
    # avg_dist_avg_asynch_W, stdev_dist_avg_asynch_W, errors_asynch_W, trans_asynch_W = dist_avg_asynch_W(rand_geo_gr, TOL)
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
    avg_pdmm_async, stdev_pdmm_async, error_pdmm_async, trans_pdmm_async = pdmm_async(rand_geo_gr, TOL)    
    plot_single_error(error_pdmm_async, "PDMM Asynch")

    
    plot_multiple_pairs(((errors_asynch_noW, "Asynch Dist Avg no W"), 
                        (error_rand_goss_noW, "Random Gossip no W"),
                        (error_pdmm_synch, "PDMM Synch"),
                        (error_pdmm_async, "PDMM Asynch")))
    
    '''
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    transmissions_synch = []
    for c in c_values:
        print("Calculating pdmm for c = ", c)
        avg, stdev, error, trans = pdmm_synch(rand_geo_gr, TOL, c)
        transmissions_synch.append((c, trans))
    plot_c_transmissions(transmissions_synch, "PDMM Synch", TOL)

    transmissions_asynch = []
    for c in c_values:
        print("Calculating pdmm for c = ", c)
        avg, stdev, error, trans = pdmm_async(rand_geo_gr, TOL, c)
        transmissions_asynch.append((c, trans))
    plot_c_transmissions(transmissions_asynch, "PDMM Asynch", TOL)
    '''

if __name__ == "__main__":
    main()
