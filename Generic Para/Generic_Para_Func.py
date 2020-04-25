#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adityapandey
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

def generic_para_func(x):
    return 0.65*x*x + 0.5*x
    
    
np_generic_para_func = np.vectorize(generic_para_func)

def d_generic_para_func(x):
    return 1.3*x + 0.5
    
np_d_generic_para_func = np.vectorize(d_generic_para_func)

np_d_generic_para_func_32 = lambda x: np_d_generic_para_func(x).astype(np.float32)

def tf_d_generic_para_func(x, name=None):
    with tf.name_scope(name, "d_generic_paraFunc", [x]) as name:
        y = tf.py_func(np_d_generic_para_func_32,
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
    
def generic_para_funcgrad(op, grad):
    x = op.inputs[0]
    
    n_gr = tf_d_generic_para_func(x)
    return grad * n_gr

np_generic_para_func_32 = lambda x: np_generic_para_func(x).astype(np.float32)

def tf_generic_para_func(x, name=None):

    with tf.name_scope(name, "generic_paraFunc", [x]) as name:
        y = py_func(np_generic_para_func_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=generic_para_funcgrad)
        return y[0]
