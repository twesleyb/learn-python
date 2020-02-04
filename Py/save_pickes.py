#!/usr/bin/env python3

import igraph as ig
import leidenalg as la
import pickle 

G = ig.Graph.Famous('Zachary')
optimiser = la.Optimiser()
profile = optimiser.resolution_profile(G,la.CPMVertexPartition, resolution_range=(0,1))

# Get key results.
results = {
        'Modularity' : [partition.modularity for partition in profile],
        'Membership' : [partition.membership for partition in profile],
        'Summary'    : [partition.summary for partition in profile],
        'Resolution' : [partition.resolution_parameter for partition in profile]}

# Function to save pickled object.
def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Save to file.
for obj in results:
    save_object(obj, 'profile_' + str(obj) + '.pkl')
