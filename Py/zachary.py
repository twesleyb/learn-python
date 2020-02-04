#!/usr/bin/env python3

import igraph as ig
import leidenalg as la

G = ig.Graph.Famous('Zachary')

# Try adding edge weight!
from numpy import random
edges = G.get_edgelist()
w = random.rand(len(edges))

G.es['weight'] = w
G.is_weighted()

# Use CPM to find partition of weighted graph.
p = la.find_partition(G, la.CPMVertexPartition, weights = 'weight', resolution_parameter = 0.05)


#############

optimiser = la.Optimiser()
profile = optimiser.resolution_profile(G, 
        la.CPMVertexPartition, 
        resolution_range=(1,1),
        number_iterations = 1)

range(0,1)


p = la.find_partition(G, la.CPMVertexPartition, resolution_parameter = 0)

len(profile)

k = [len(partition) for partition in profile]
r = [partition.resolution_parameter for partition in profile]
q = [partition.q for partition in profile]

profile[7].sizes()

profile[7].summary()

profile[7].q

