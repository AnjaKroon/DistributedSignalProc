# AUTHOR: Anja Kroon

# Code for distributed SP project
# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

from utils import generate_rgg, generate_measurements, vector_to_dict
from algorithms import dist_avg_synch, dist_avg_asynch_W, dist_avg_asynch_noW, random_gossip_noW, pdmm_synch, pdmm_async, random_gossip_TF, pdmm_async_tf, random_gossip_dropadd, pdmm_asynch_dropadd, dist_avg_asynch_noW_tf
from visualization import plot_single_error, plot_multiple_pairs, plot_c_transmissions, plot_rgg_nodes
import numpy as np

# MANUAL
# NODES = 100
# RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01

NODES = 200
RAD = np.sqrt(np.log(2*NODES) / NODES)
print(RAD)

#AUTO
# PROB_CONN = 0.9
# NODES = int(np.ceil(np.sqrt(1 / (1 - PROB_CONN))))

DIM = 2
TOL = 10**-12
FAILURE_RATE = 0.25

def main():

    temps = generate_measurements(NODES)
    dict_temps = vector_to_dict(temps)
    rand_geo_gr = generate_rgg(NODES, RAD, DIM, dict_temps)

    plot_rgg_nodes(rand_geo_gr, "Temperature Sensors")
    
    
    # SYNCH DIST AVG
    avg_dist_avg_synch, stdev_dist_avg_synch, errors_synch, trans_synch = dist_avg_synch(rand_geo_gr, TOL)
    # plot_single_error(errors_synch, "Synch Dist Avg")

    # ASYNCH DIST AVG
    avg_dist_avg_asynch_W, stdev_dist_avg_asynch_W, errors_asynch_W, trans_asynch_W = dist_avg_asynch_W(rand_geo_gr, TOL)
    # plot_single_error(errors_asynch_W, "Asynch Dist Avg (W)")
    avg_asynch_noW, stdev_asynch_noW, errors_asynch_noW, trans_asynch_noW = dist_avg_asynch_noW(rand_geo_gr, TOL)
    # plot_single_error(errors_asynch_noW, "Asynch Dist Avg (No W)")

    avg_DA_TF_zero, stdev_DA_TF_zero, err_DA_TF_zero, trans_DA_TF_zero = dist_avg_asynch_noW_tf(rand_geo_gr, TOL, FAILURE_RATE=0.0)
    avg_DA_TF_zero, stdev_DA_TF_25, err_DA_TF_25, trans_DA_TF_25 = dist_avg_asynch_noW_tf(rand_geo_gr, TOL, FAILURE_RATE=0.25)
    avg_DA_TF_zero, stdev_DA_TF_50, err_DA_TF_50, trans_DA_TF_50 = dist_avg_asynch_noW_tf(rand_geo_gr, TOL, FAILURE_RATE=0.50)
    avg_DA_TF_zero, stdev_DA_TF_75, err_DA_TF_75, trans_DA_TF_75 = dist_avg_asynch_noW_tf(rand_geo_gr, TOL, FAILURE_RATE=0.75)

    # RANDOM GOSSIP
    avg_rand_goss, stdev_rand_goss_noW, error_rand_goss_noW, trans_rand_goss_noW = random_gossip_noW(rand_geo_gr, TOL)
    # plot_single_error(error_rand_goss_noW, "Random Gossip no W")

    plot_multiple_pairs(((err_DA_TF_zero, "Dist Avg 0%"),
                        (err_DA_TF_25, "Dist Avg 25%"),
                        (err_DA_TF_50, "Dist Avg 50%"),
                        (err_DA_TF_75, "Dist Avg 75%"), 
                        (error_rand_goss_noW, "Random Gossip 0% ")), "Dist Avg with Transmission Failure")
    
    # PDMM with c = 0.4 and failure rate = 0.0
    avg_pdmm_synch, stdev_pdmm_synch, error_pdmm_synch, trans_pdmm_synch = pdmm_synch(rand_geo_gr, TOL)
    # plot_single_error(error_pdmm_synch, "PDMM Synch")
    

    avg_pdmm_async, stdev_pdmm_async, error_pdmm_async, trans_pdmm_async = pdmm_async(rand_geo_gr, TOL)
    # plot_single_error(error_pdmm_async, "PDMM Asynch")

    
    # COMPARING ALL ALGORITHMS UNDER IDEAL SCENARIOS
    plot_multiple_pairs(((errors_asynch_noW, "Dist Avg Asynch"),
                        (error_rand_goss_noW, "Random Gossip"),
                        (error_pdmm_async, "PDMM Asynch")), "All Algorithms under Ideal Scenario (Nodes: " + str(NODES) + ")")

    # CHOICE OF C IN PDMM
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

    # TRANSMISSION FAILURES IN RANDOM GOSSIP
    avg_rg_tf, stdev_rg_tf, error_rg_tf, trans_rg_tf = random_gossip_TF(rand_geo_gr, TOL, 0.1)
    # plot_single_error(error_rg_tf, "Random Gossip with Transmission Failure 10%")

    avg_rg_tf0,  stdev_rg_tf0,  error_rg_tf0,  trans_rg_tf0  = random_gossip_TF(rand_geo_gr, TOL, 0.0)
    avg_rg_tf25, stdev_rg_tf25, error_rg_tf25, trans_rg_tf25 = random_gossip_TF(rand_geo_gr, TOL, 0.25)
    avg_rg_tf50, stdev_rg_tf50, error_rg_tf50, trans_rg_tf50 = random_gossip_TF(rand_geo_gr, TOL, 0.50)
    avg_rg_tf75, stdev_rg_tf75, error_rg_tf75, trans_rg_tf75 = random_gossip_TF(rand_geo_gr, TOL, 0.75)
    
    plot_multiple_pairs(((error_rg_tf0, "Random Gossip with (0%)"),
                        (error_rg_tf25, "Random Gossip with (25%)"),
                        (error_rg_tf50, "Random Gossip with (50%)"),
                        (error_rg_tf75, "Random Gossip with (75%)")), "Random Gossip with Transmission Failure (Nodes: " + str(NODES) + ")")
    
    # TRANSMISSION FAILURES IN RANDOM GOSSIP
    avg_pdmm_tf, stdev_pdmm_tf, error_pdmm_tf, trans_pdmm_tf = pdmm_async_tf(rand_geo_gr, TOL)

    avg_pdmm_tf0,  stdev_pdmm_tf0,  error_pdmm_tf0,  trans_pdmm_tf0 = pdmm_async_tf(rand_geo_gr, TOL, c=0.4, FAILURE_RATE=0.0)
    avg_pdmm_tf25, stdev_pdmm_tf25, error_pdmm_tf25, trans_pdmm_tf25 = pdmm_async_tf(rand_geo_gr, TOL,  c=0.4, FAILURE_RATE=0.25)
    avg_pdmm_tf50, stdev_pdmm_tf50, error_pdmm_tf50, trans_pdmm_tf50 = pdmm_async_tf(rand_geo_gr, TOL,  c=0.4, FAILURE_RATE=0.50)
    avg_pdmm_tf75, stdev_pdmm_tf75, error_pdmm_tf75, trans_pdmm_tf75 = pdmm_async_tf(rand_geo_gr, TOL,  c=0.4, FAILURE_RATE=0.75)
    
    plot_multiple_pairs(((error_pdmm_tf0, "PDMM with (0%)"),
                        (error_pdmm_tf25, "PDMM with (25%)"),
                        (error_pdmm_tf50, "PDMM with (50%)"),
                        (error_pdmm_tf75, "PDMM with (75%)")), "PDMM with Transmission Failure (Nodes: " + str(NODES) + ")")
    
    # TESTING BULK DROP
    # only have drop or add > 0, not both
    avg_rg_tf_drop_bulk, stdev_rg_tf_drop_bulk, error_rg_tf_drop_bulk, trans_rg_tf_drop_bulk= random_gossip_dropadd(rand_geo_gr, TOL, DROP_RATE=0.1, ADD_RATE=0.0, type="bulk")
    # plot_single_error(error_rg_tf_dropadd, ("Random Gossip with Dropping "+str(dr)+" and Adding "+str(fr)))
    plot_multiple_pairs(((error_rand_goss_noW, "Random Gossip (Drop=0%)"),
                        (error_rg_tf_drop_bulk, "Random Gossip (Drop=" + str(0.1*100) + "%)")), "Random Gossip: Drop: " + str(0.1*100) + "% Type: Bulk ")
    

    # TESTING SEQUENTIAL DROP
    # only have drop or add > 0, not both
    avg_rg_tf_drop_seq, stdev_rg_tf_drop_seq, error_rg_tf_drop_seq, trans_rg_tf_drop_seq= random_gossip_dropadd(rand_geo_gr, TOL, DROP_RATE=0.1, ADD_RATE=0.0, type="seq")
    plot_multiple_pairs(((error_rand_goss_noW, "Random Gossip (Drop=0%)"),
                        (error_rg_tf_drop_seq, "Random Gossip (Drop=" + str(0.1*100) + "%)")), "Random Gossip: Drop: " + str(0.1*100) + "% Type: Sequential ")
    
    # TESTING BULK ADD
    avg_rg_tf_add_bulk, stdev_rg_tf_add_bulk, error_rg_tf_add_bulk, trans_rg_tf_add_bulk= random_gossip_dropadd(rand_geo_gr, TOL, DROP_RATE=0.0, ADD_RATE=0.1, type="bulk")
    plot_multiple_pairs(((error_rand_goss_noW, "Random Gossip (Add=0%)"),
                        (error_rg_tf_add_bulk, "Random Gossip (Add=" + str(0.1*100) + "%)")), "Random Gossip: Add: " + str(0.1*100) + "% Type: Bulk ")
    

    # TESTING SEQUENTIAL ADD
    avg_rg_tf_add_seq, stdev_rg_tf_add_seq, error_rg_tf_add_seq, trans_rg_tf_add_seq= random_gossip_dropadd(rand_geo_gr, TOL, DROP_RATE=0.0, ADD_RATE=0.1, type="seq")
    plot_multiple_pairs(((error_rand_goss_noW, "Random Gossip (Add=0%)"),
                        (error_rg_tf_add_seq, "Random Gossip (Add=" + str(0.1*100) + "%)")), "Random Gossip: Add: " + str(0.1*100) + "% Type: Sequential ")
         
    # TESTING PDMM BULK DROP (NEW METHOD)
    avg_pdmm_drop_bulk, stdev_pdmm_drop_bulk, error_pdmm_drop, trans_pdmm_drop_bulk = pdmm_asynch_dropadd(rand_geo_gr, TOL, c=0.4, DROP_RATE=0.5, ADD_RATE=0.0)
    plot_multiple_pairs( ((error_pdmm_async, "PDMM Asynch"), 
                        (error_pdmm_drop, "PDMM Asynch (Drop=50%)")), "Comparing PDMM: Baseline & Bulk Drop")

    # TESTING PDMM BULK ADD (NEW METHOD)
    avg_pdmm_add_bulk, stdev_pdmm__add_bulk, error_pdmm_add, trans_pdmm_add_bulk = pdmm_asynch_dropadd(rand_geo_gr, TOL, c=0.4, DROP_RATE=0.0, ADD_RATE=0.5)
    plot_multiple_pairs( ((error_pdmm_async, "PDMM Asynch"), 
                        (error_pdmm_add, "PDMM Asynch (Add=50%)")), "Comparing PDMM: Baseline & Bulk Add")
    
    
    

if __name__ == "__main__":
    main()
