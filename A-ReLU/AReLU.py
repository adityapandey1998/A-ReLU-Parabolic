#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:31:34 2019

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
        return 0
    else: 
        return ARelu_k*np.power(x, ARelu_n)
    
np_ARelu = np.vectorize(ARelu)

def d_ARelu(x):
    global ARelu_k, ARelu_n
    if x<=0:
        return 0
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
    
'''

with tf.Session() as sess:
    
    x = tf.constant([0.25,1.7])
    x = tf.constant([[0.2,0.7],[1.2,-8.7]])
    y = tf_ARelu(x)
    tf.initialize_all_variables().run()
    
    set_kn(1, 0.5)
    print(x.eval(), y.eval() , tf.gradients(y, [x])[0].eval())
    

#Test Code

set_kn(0.9, 1.2)

X_train = np.random.randn(50,1)
Y_train = X_train* X_train + 2* X_train +3

lr = 0.001
epochs = 300

#Two placeholders for training
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

size_input = 1
size_output = 1
size_hidden1 = 2
size_hidden2 = 6
size_hidden3 = 18

#Weights and biases
Wh1 = tf.Variable(tf.random_normal([size_input,size_hidden1]))
bh1 = tf.Variable(tf.random_normal([1,size_hidden1]))

Wh2 = tf.Variable(tf.random_normal([size_hidden1,size_hidden2]))
bh2 = tf.Variable(tf.random_normal([1,size_hidden2]))

Wy = tf.Variable(tf.random_normal([size_hidden2,size_output]))
by = tf.Variable(tf.random_normal([1,size_output]))

a1 = tf.matmul(X,Wh1)+bh1
lh1 = tf_ARelu(a1) 
#lh1 = tf.nn.relu(a1) #Not tanh as it limits between -1 and +1

#pred1 = tf.matmul(lh1,Wh2) + bh2
a2 = tf.matmul(lh1,Wh2)+bh2
lh2 = tf.nn.relu(a2) 
lh2 = tf_ARelu(a2)

pred = tf.matmul(lh2,Wy) + by

#Variables for training
init = tf.global_variables_initializer()

#Calculate predicted output

#Define the loss function
#loss = tf.reduce_sum(tf.pow((Y-pred),2.0)) / (2.0*len(X_train))
loss = tf.pow(Y-pred,2.0) / 2.0

#Create the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


#Execute
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        count = 0
        while count<=49:
            sess.run(opt, feed_dict={X:X_train[[count]], Y:Y_train[[count]]} ) 
            count+=1
            
        if epoch % 50 ==0:
            print("Epoch:",epoch)
            #Just print the loss function as rms or something, won't work with loss.eval()
            #print("loss:", loss.eval({X:X_train[count], Y:Y_train[count] })) 
 
    test = np.array([1.1,1.2,1.3,1.25,0.8,0.9])
    X_test = [[1.1],[1.2],[1.3],[1.25],[0.8],[0.9]]
    Y_test = test* test+ 2* test+3
    Y_pred = sess.run(pred,feed_dict={X:X_test})
    

'''