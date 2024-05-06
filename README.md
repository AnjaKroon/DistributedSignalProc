# Distributed Signal Processing

This repository contains code for distributed signal processing algorithms solving the average consensus problem implemented in Python. 

The primary algorithms investigated are:
* Distributed asynchronous averaging
* Randomized gossip
* Primal Dual Method of Multipliers (PDMM)

There are two versions of the codebase -- Anja's and Frank's.

Both versions implement distributed averaging (synchronous, asynchronous), random gossip, and primal-dual method of multipliers (synchronous, asynchronous). Both codebases also provide the relevant functions to plot the random geometric graphs (RGGs) used to describe the problem setting. Both implementations also produce plots for the primary purposes of comparing algorithms. However, there are some additional experiments conducted such as finding the optimal c hyperparameter in PDMM and comparing the synchronous and asynchornous implementations.

The final report is available in: `Distributed_Final_Report_FinalVersion.pdf`.

## To run Anja's

Install dependencies available in `requirements.txt`. Run the main.py file to recreate results from the report including additional experiments. To obtain only certain results, comment out the other function calls in `main()`.

## To run Frank's

Install dependencies available in `requirements.txt`. Run cells of `frank.ipynb` to recreate results, or view existing results.
