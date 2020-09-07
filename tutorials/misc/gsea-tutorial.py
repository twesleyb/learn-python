# An example to run Prerank using gseapy prerank module
#$ gseapy prerank -r gsea_data.rnk -g gene_sets.gmt -o test

# An example to run Prerank using gseapy prerank module
#$ gseapy prerank -r gsea_data.rnk -g gene_sets.gmt -o test

import pandas as pd

rnk = pd.read_csv("exampleRanks.rnk", header=None, sep="\\,")
rnk.head()


import gseapy as gp

re_res = gp.prerank(rnk=rnk, gene_sets='KEGG_2016',
                             processes=4,
                             permutation_num=100, # reduce number to speed up
                             outdir='test')
