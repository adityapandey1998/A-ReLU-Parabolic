#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:23:48 2019

@author: adityapandey
"""
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

ARelu_k=1
ARelu_n=1

def set_kn(k, n):
    global ARelu_k, ARelu_n
    ARelu_k = k
    ARelu_n = n

def ARelu(x):
    global ARelu_k, ARelu_n
    if x<=0:
        return -0.01 * ARelu_k * np.power(abs(x), ARelu_n)
    else: 
        return ARelu_k * np.power(x, ARelu_n)
    
np_ARelu = np.vectorize(ARelu)

def d_ARelu(x):
    global ARelu_k, ARelu_n
    if x<=0:
        return -0.01 * ARelu_n * ARelu_k * np.power(abs(x), ARelu_n)
    else: 
        return ARelu_n * ARelu_k * np.power(x, ARelu_n-1)
    
np_d_ARelu = np.vectorize(d_ARelu)

np_d_ARelu_32 = lambda x: np_d_ARelu(x).astype(np.float32)

def tf_d_ARelu(x, name=None):
    with tf.name_scope(name, "d_ARelu", [x]) as name:
        y = tf.py_func(np_d_ARelu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
def ARelugrad(op, grad):
    x = op.inputs[0]
    
    n_gr = tf_d_ARelu(x)
    return grad * n_gr

np_ARelu_32 = lambda x: np_ARelu(x).astype(np.float32)

def tf_ARelu(x, name=None):

    with tf.name_scope(name, "ARelu", [x]) as name:
        y = py_func(np_ARelu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=ARelugrad)
        return y[0]
    
