#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:52:39 2020

@author: adityapandey
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

def abs_func(x):
    if x<0:
        return -x
    return x
    
    
np_abs_func = np.vectorize(abs_func)

def d_abs_func(x):
    if x<0:
        return -1
    if x==0:
        return 0
    return 1
    
np_d_abs_func = np.vectorize(d_abs_func)

np_d_abs_func_32 = lambda x: np_d_abs_func(x).astype(np.float32)

def tf_d_abs_func(x, name=None):
    with tf.name_scope(name, "d_absFunc", [x]) as name:
        y = tf.py_func(np_d_abs_func_32,
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
    
def abs_funcgrad(op, grad):
    x = op.inputs[0]
    
    n_gr = tf_d_abs_func(x)
    return grad * n_gr

np_abs_func_32 = lambda x: np_abs_func(x).astype(np.float32)

def tf_abs_func(x, name=None):

    with tf.name_scope(name, "absFunc", [x]) as name:
        y = py_func(np_abs_func_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=abs_funcgrad)
        return y[0]
