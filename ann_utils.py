import tensorflow as tf
import math
import os

def create_weights_and_biases(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 2):
        weight = create_weight(layer_sizes[i], layer_sizes[i+1])
        bias = tf.Variable(tf.ones([layer_sizes[i+1]])/layer_sizes[0])
        weights.append(weight)
        biases.append(bias)

    weight = create_weight(layer_sizes[-2], layer_sizes[-1])
    bias = tf.Variable(tf.zeros([layer_sizes[-1]])/layer_sizes[0])
    weights.append(weight)
    biases.append(bias)
    return weights, biases

def link_weights_and_biases(inputs, weights, biases, activation=None, pkeep=None):
    Y = tf.matmul(inputs, weights[0]) + biases[0]
    for i in range(1, len(weights)):
        if not activation == None:
            Y = activation(Y)
        if not pkeep == None:
            Y = tf.nn.dropout(Y, pkeep)
        Y = tf.matmul(Y, weights[i]) + biases[i]
    return Y

def create_weight(in_size, out_size):
    return tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))

def l2_regularize_weights(weights):
    loss = tf.nn.l2_loss(weights[-1])
    for i in range(len(weights) - 1):
        weight = weights[i]
        loss = loss + tf.nn.l2_loss(weight)
    return loss

def minibatch(data, start, length):
    start_index = start % len(data)
    end_index = start_index + length
    nloops = math.ceil(end_index / len(data))
    looped_data = nloops * data
    return looped_data[start_index:end_index]
