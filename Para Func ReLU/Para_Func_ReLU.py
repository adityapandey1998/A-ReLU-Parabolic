#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:23:07 2020

@author: adityapandey
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

Para_Func_k = 1

def set_k(k):
    global Para_Func_k
    Para_Func_k = k

def para_func(x):
    return Para_Func_k * np.power(x, 2)
    
    
np_para_func = np.vectorize(para_func)

def d_para_func(x):
    return 2 * Para_Func_k * x
    
np_d_para_func = np.vectorize(d_para_func)

np_d_para_func_32 = lambda x: np_d_para_func(x).astype(np.float32)

def tf_d_para_func(x, name=None):
    with tf.name_scope(name, "d_ParabolicFunc", [x]) as name:
        y = tf.py_func(np_d_para_func_32,
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
    
def para_funcgrad(op, grad):
    x = op.inputs[0]
    
    n_gr = tf_d_para_func(x)
    return grad * n_gr

np_para_func_32 = lambda x: np_para_func(x).astype(np.float32)

def tf_para_func(x, name=None):

    with tf.name_scope(name, "ParabolicFunc", [x]) as name:
        y = py_func(np_para_func_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=para_funcgrad)
        return y[0]
