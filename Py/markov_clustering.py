#!/usr/bin/env python3

from markov_clustering import run_mcl
from markov_clustering import get_clusters
from markov_clustering import modularity
import networkx as nx
import random

# number of nodes to use
numnodes = 200

# generate random positions as a dictionary where the key is the node id and the value
# is a tuple containing 2D coordinates
positions = {i:(random.random() * 2 - 1, random.random() * 2 - 1) for i in range(numnodes)}

# use networkx to generate the graph
network = nx.random_geometric_graph(numnodes, 0.3, pos=positions)

# then get the adjacency matrix (in sparse form)
matrix = nx.to_scipy_sparse_matrix(network)

import markov_clustering as mcl
import networkx as nx

A = g.get_adjacency()
A = np.array(A.data)
adj_matrix = nx.to_numpy_matrix(G)
adj_matrix = nx.to_numpy_matrix(G)
res = mcl.run_mcl(adj_matrix)
clusters = mcl.get_clusters(res)

# run the MCL algorithm on the adjacency matrix 
result = run_mcl(matrix)           # run MCL with default parameters
clusters = get_clusters(result)    # get clusters

# If the clustering is too fine for your taste, reducing the MCL inflation parameter to 1.4 (from the default of 2)
# will result in coarser clustering. e.g.
result = run_mcl(matrix, inflation=1.4)
clusters = get_clusters(result)

## Choosing Hyperparameters

# Choosing appropriate values for hyperparameters (e.g. cluster inflation/expansion parameters) can be difficult.  
# To assist with the evaluation of the clustering quality, we include an implementation of the modularity measure.
# Refer to 'Malliaros et al., Physics Reports 533.4 (2013): 95-142' for a detailed description.  

# Briefly, the modularity (Q) can be considered to be the fraction of graph edges which belong to a cluster 
# minus the fraction expected due to random chance, where the value of Q lies in the range [-1, 1]. High, positive
# Q values suggest higher clustering quality.  

# We can use the modularity measure to optimize the clustering parameters. In the following example,
# we will determine the modularity for a range of cluster inflation values, allowing us to pick the best 
# cluster inflation value for the given graph.

# perform clustering using different inflation values from 1.5 and 2.5
# for each clustering run, calculate the modularity
for inflation in [i / 10 for i in range(15, 26)]:
    result = run_mcl(matrix, inflation=inflation)
    clusters = get_clusters(result)
    Q = modularity(matrix=result, clusters=clusters)
    print("inflation:", inflation, "modularity:", Q)

# From the output, we see that an inflation value of 2.1 gives the highest modularity score,
# so we will use that as our final cluster inflation parameter.

# cluster using the optimized cluster inflation value
result = run_mcl(matrix, inflation=2.1)
clusters = get_clusters(result)
