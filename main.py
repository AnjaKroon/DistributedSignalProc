# AUTHOR: Anja Kroon

# Code for distributed SP project
# Objective: Compute the average value of the measurement data
# Need to do random gossip algorithm, and another second decentralized asynchonous algorithm

from utils import generate_rgg, generate_measurements, vector_to_dict
from algorithms import dist_avg_synch, dist_avg_asynch_W, dist_avg_asynch_noW, random_gossip_noW
from visualization import plot_single_error, plot_multiple_pairs

NODES = 100
RAD = 0.3067   # 100 km, radius is 1 km 1/100 = 0.01
DIM = 2
TOL = 0.000000001

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
    

if __name__ == "__main__":
    main()
