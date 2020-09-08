## imports -------------------------------------------------

from __future__ import division
from __future__ import print_function

# ignore tf INFO and WARNING messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

import requests

import time

import numpy as np

import scipy.sparse as sp

import networkx as nx

import tensorflow as tf

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score


## function ismember -----------------------

def ismember(a, b):
    '''
    '''
    rows_close = np.all((a - b[:, None]) == 0, axis=-1)
    return np.any(rows_close)


## function sigmoid -----------------------

def sigmoid(x):
    '''
    '''
    return 1 / (1 + np.exp(-x))


## function download_data -----------------------

def download_data():
    '''
    download the data from stanford.edu
    @import requests
    '''
    URL='http://snap.stanford.edu/deepnetbio-ismb/ipynb/yeast.edgelist'

    if os.path.exists('yeast.edgelist'):
        warnings.warn("Overwriting existing file: 'yeast.edgelist'.")

    response = requests.get(URL)

    os.path.basename(URL)

    with  open('yeast.edgelist','wb') as f:
        f.write(response.content)


## function load_data -----------------------

def load_data():
    '''
    load yeast.edgelist as sparse matrix
    @import networkx as nx
    '''
    g = nx.read_edgelist('yeast.edgelist')
    adjm = nx.adjacency_matrix(g)
    return adjm

## function weight_variable_glorot -----------------------

def weight_variable_glorot(input_dim,
        output_dim, name=""):
    '''
    @import numpy as np
    @import tensorflow as tf
    '''
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # random_uniform is deprecated, instead use:
    initial = tf.compat.v1.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

## function dropout_sparse --------------------------------

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    '''
    @import tensorflow as tf
    '''
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor),
            dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)

## function sparse_to_tuple -------------------------------

def sparse_to_tuple(sparse_mx):
    '''
    coerce sparse adjm to tuple coordinants
    @import scipy.sparse as sp
    @import numpy as np
    '''
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,
        sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

## function preprocess_graph ------------------------------

def preprocess_graph(adj):
    '''
    normalize the ppi matrix - 1/sqrt ... ?
    @local sparse_to_tuple
    @import scipy.sparse as sp
    '''
    adj = sp.coo_matrix(adj) # sp matrix coordinate format
    adj_ = adj + sp.eye(adj.shape[0]) # diag = 1
    rowsum = np.array(adj_.sum(1)) # node degree
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

## function construct_feed_dict ---------------------------

def construct_feed_dict(adj_normalized,
        adj, features, placeholders):
    '''
    '''
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

## function mask_test_edges -------------------------------

def mask_test_edges(adj):
    '''
    Function to build test set with 2% positive links
    Remove diagonal elements
    @local sparse_to_tuple
    @import scipy.sparse as sp
    @import numpy as np
    '''
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :],[0]),
            shape=adj.shape)
    adj.eliminate_zeros()
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 50.)) #2% =130
    num_val = int(np.floor(edges.shape[0] / 50.))
    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(list(all_edge_idx)) # added: list()
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges,
            np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    #
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        '''
        loop that does something unitl
        '''
        n_rnd = len(test_edges) - len(test_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])
    #EOL
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        '''
        loop that does something else
        '''
        n_rnd = len(val_edges) - len(val_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])
    #EOL
    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0],
        train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return(adj_train, train_edges, val_edges,
            val_edges_false, test_edges, test_edges_false)
#EOF


## function get_roc_score --------------------------------

def get_roc_score(edges_pos, edges_neg):
    '''
    # note: sess = tf session
    @local sigmoid
    @import numpy as np
    @import tensorflow as tf
    @import from sklearn.metrics import roc_auc_score
    @import from sklearn.metrics import average_precision_score
    '''
    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    #
    #
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    #
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])
    #
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)),
        np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    #
    return roc_score, ap_score
#EOF
