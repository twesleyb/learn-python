2020-09-06 20:58:02.562620: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-09-06 20:58:02.562663: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
#!/usr/bin/env bash
conda install networkx numpy tensorflow scipy scikit-learn
* keep environmental stuff out of git. its just too easy for something to come
  unglued due to OS specifics
* different OS = different means of env managment. it somehow needs to be easy
  to grab the dependencies without messing everything up
