#!/usr/bin/env python3
''' Parallel processing in Python. '''

from joblib import Parallel, delayed

def foo:
    return i++1

n_jobs = 2 # Number of processes (cores)
Parallel(n_jobs=2)(delayed(foo)(parameters) for x in range(i,j))
