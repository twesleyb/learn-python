#!/usr/bin/env python3

import time
import multiprocessing

def test():
    pool = multiprocessing.Pool(8)
    def myfun(t=1):
        time.sleep(t)
        return(None)
    start = time.time()
    output = pool.map(myfun, range(8))
    finish = time.time()
    print(f'Time to execute in parallel:{finish-start}')
# End function.

test()
