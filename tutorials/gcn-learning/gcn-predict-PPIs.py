#!/usr/bin/env python3
# coding: utf-8

'''
Graph Convolutional Prediction of PPIs
from: http://snap.stanford.edu/deepnetbio-ismb/

notes:

# utilizes the following functions in Py/local.py:
* def download_data
* def load_data
* def weight_variable_glorot
* def dropout_sparse
* def sparse_to_tuple
* def preprocess_graph
* def construct_feed_dict
* def mask_test_edges
* def ismember
* def get_roc_score
* def sigmoid


# utilzes the following clases defined below:
* class GraphConvolution
* class GraphConvolutionSparse
* class InnerProductDecoder
* class GCNModel
* class Optimizer

'''

# input --------------------------------------------------

# 'yeast.edgelist' - PPIs in a simple format: protA protB
# source: http://snap.stanford.edu/deepnetbio-ismb/ipynb/yeast.edgelist

# imports -------------------------------------------------

from __future__ import division
from __future__ import print_function

# ignore tf INFO and WARNING messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests

import time

import numpy as np

import scipy.sparse as sp

import networkx as nx

import tensorflow as tf

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score

# local imports
from Py import local

# options -------------------------------------------------

# set seed for reproducibility
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings - use of tf.app.flags is outdated,
# use tf.compat module
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01,
        'Initial learning rate.')
flags.DEFINE_integer('epochs', 20,
        'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32,
        'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16,
        'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1,
        'Dropout rate (1 - keep probability).')

## class GraphConvolution ---------------------------------

class GraphConvolution():
    '''
    Basic graph convolution layer for undirected
    graph without edge labels.
    @local weight_variable_glorot
    @import tensorflow as tf
    '''
    def __init__(self,
            input_dim,
            output_dim,
            adj,
            name,
            dropout=0.,
            act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = local.weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
    #
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs
#EOC


## class GraphConvolutionSparse --------------------------

class GraphConvolutionSparse():
    '''
    Graph convolution layer for sparse inputs.
    @import tensorflow as tf
    '''
    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = local.weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
    #
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs
#EOC


## class InnerProductDecoder ------------------------------

class InnerProductDecoder():
    '''
    Decoder model layer for link prediction
    '''
    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act
    #
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            x = tf.transpose(inputs)
            x = tf.matmul(inputs, x)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
#EOC

## class GCNModel -----------------------------------------

class GCNModel():
    '''
    what do i do?
    @local GraphConvolution # embedings
    @local GraphConvolutionSparse # hidden layer
    @local InnerProductDecoder # reconstructions
    @import tensorflow as tf
    '''
    def __init__(self, placeholders, num_features, features_nonzero, name):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        with tf.variable_scope(self.name):
            self.build()
    #
    def build(self):
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout)(self.inputs)
    #
        self.embeddings = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden1)
    #
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden2,
            act=lambda x: x)(self.embeddings)
#EOC


## class Optimizer ---------------------------------------
# Define the GCN Optimizer class

class Optimizer():
    '''
    notes:
    * uses the ADAM optimizer
    @import tensorflow as tf
    '''
    def __init__(self, preds, labels,
            num_nodes, num_edges):
        pos_weight = float(num_nodes**2 - num_edges) / num_edges
        norm = num_nodes**2 / float((num_nodes**2 - num_edges) * 2)
    #
        preds_sub = preds
        labels_sub = labels
    #
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
    #
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


## main --------------------------------------------------
# Train the model and and then evaluate its accuracy on a
# Test Set of Protein-Protein Interactions

# Given a training set of protein-protein interactions in
# yeast (S. cerevisiae), our goal is to take these
# interactions and train a GCN model that can predict new
# protein-protein interactions. That is, we would like to
# predict new edges in the yeast PPI network.

if __name__ == "__main__" :

    # download the data
    local.download_data()

    # create sparse adjacency matrix
    adjm = local.load_data()

    # summarize the network
    num_nodes = adjm.shape[0]
    num_edges = adjm.sum()
    print("\nLoaded yeast PPI graph with:")
    print("\tNumber of nodes: {}".format(num_nodes))
    print("\tNumber of edges: {}".format(num_edges))

    # coerce graph to tuple describing the graphs features
    #import scipy.sparse as sp
    features = local.sparse_to_tuple(sp.identity(num_nodes))

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Store original adjacency matrix
    # (without diagonal entries) for later
    #import numpy as np
    adjm_orig = adjm - sp.dia_matrix((adjm.diagonal()[np.newaxis, :], [0]), shape=adjm.shape)
    adjm_orig.eliminate_zeros()

    # generate the test and train datasets
    # NOTE: this takes some time
    # i think this takes a bunch of time bc its looping to create the test/train data in some sort of random process
    adjm_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = local.mask_test_edges(adjm)

    # Trying to figure out what's what
    # adj_train - sparse 6526 x 6526 matrix - 1,018,554 edges
    #             this is the input matrix?
    # train_edges - edges list, array len 509,277
    # val_edges - ?? array len 10,609 (2%?)
    # val_edges_false - array len 10,609
    # test_edges - array len 10,609
    # test_edges_false array len 10,609

    adjm = adjm_train

    # do some normalization
    adjm_norm = local.preprocess_graph(adjm)

    # Define placeholders
    # tf.sparse_placehoder is deprecated, use compat module
    #import tensorflow as tf
    # error: sparse_placeholder` is not compatible with
    #        eager execution
    # using compat.v1.float32 does not seem to fix

    # FIXME: breaks here?
    from tensorflow.compat.v1 import float32 as f32
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(f32),
        'adj': tf.v1.compat.v1.sparse_placeholder(f32),
        'adj_orig': tf.compat.v1.sparse_placeholder(f32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    # Create model
    model = GCNModel(placeholders,
            num_features,
            features_nonzero,
            name='yeast_gcn')

    # create optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            num_nodes=num_nodes,
            num_edges=num_edges)

    # initialize tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm,
                adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # One update of parameter matrices
        _, avg_cost = sess.run([opt.opt_op, opt.cost],
                feed_dict=feed_dict)
        # Performance on validation set
        roc_curr, ap_curr = get_roc_score(val_edges,
                val_edges_false)
        # Progress report:
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(avg_cost),
              "val_roc=", "{:.5f}".format(roc_curr),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))
    #EOL
    print('Optimization Finished!')

    # Print the results:
    roc_score, ap_score = get_roc_score(test_edges,
            test_edges_false)
    print('Test ROC score: {:.5f}'.format(roc_score))
    print('Test AP score: {:.5f}'.format(ap_score))
#EOF
