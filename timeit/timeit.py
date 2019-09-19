#!/usr/bin/env python3

# The "timeit" module lets you measure the execution
# time of small bits of Python code

from ttictoc import TicToc
from time import sleep

t = TicToc()
t.tic()
sleep(2.5)
t.toc()
print(t.elapsed)
