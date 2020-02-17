#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:38:52 2019

@author: adityapandey
"""

from sklearn.utils import resample



import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import AReLU

AReLU.set_kn(0.82, 1.12)
#AReLU.set_kn(0.54, 1.3)
#print(AReLU.ARelu_k, AReLU.ARelu_n)

seed = 1
rng = np.random.RandomState(seed)

phL_EC = pd.read_csv('pHL_EC.csv')



phL_EC.drop(['Index','P. Name'], axis=1, inplace=True)
phL_EC['P. Habitable Class'] = phL_EC['P. Habitable Class'].astype(str)
phL_EC = phL_EC.loc[phL_EC['P. Habitable Class'].isin(["non-habitable", "mesoplanet", "psychroplanet"])]

df_minority1 = phL_EC[phL_EC['P. Habitable Class']=='mesoplanet']
df_minority2 = phL_EC[phL_EC['P. Habitable Class']=='psychroplanet']
df_majority = phL_EC[phL_EC['P. Habitable Class']=='non-habitable']

df_minority1_upsampled = resample(df_minority1, 
                                 replace=True,     
                                 n_samples=90,    
                                 random_state=123)

df_minority2_upsampled = resample(df_minority2, 
                                 replace=True,     
                                 n_samples=54,    
                                 random_state=123)

phL_EC = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled])

#["non-habitable", "mesoplanet", "psychroplanet"]

#phL_EC['P. Habitable Class'] = phL_EC['P. Habitable Class'].astype(str)
categories = phL_EC['P. Habitable Class']
print(categories.value_counts())
num_cat = len(categories.value_counts())

values = np.array(categories)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
labels = pd.DataFrame(onehot_encoded)


data = phL_EC.drop('P. Habitable Class', axis=1)

data.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=4, stratify=labels)
batch_size =len(X_train)

print(X_train.shape, y_train.shape,y_test.shape )


def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('X_'+dataset_name).iloc[batch_mask].values
    #print(batch_x)
    if dataset_name == 'train':
        batch_y = eval('y_'+dataset_name).iloc[batch_mask].values
        
    return batch_x, batch_y


### set all variables

# number of neurons in each layer
input_num_units = X_train.shape[1]
hidden_num_units1 = 12
hidden_num_units2 = 20
output_num_units = num_cat

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 300
batch_size = X_train.shape[0]
#batch_size = 256
learning_rate = 0.015

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units1, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
#hidden_layer1 = tf.nn.relu(hidden_layer1)
hidden_layer1 = AReLU.tf_ARelu(hidden_layer1)

'''
hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
#hidden_layer2 = tf.nn.relu(hidden_layer2)
hidden_layer2 = AReLU.tf_ARelu(hidden_layer2)
'''

output_layer = tf.add(tf.matmul(hidden_layer1, weights['output']), biases['output'])

logits = output_layer
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

#optimizer = tf.train.MomentumOptimizer(momentum = 0.01, learning_rate=learning_rate).minimize(cost)

#cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(output_layer) + (1 - y) * tf.log(1 - output_layer), axis=1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

display_step=10

pred=[]

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, epochs+1):
        batch_x, batch_y = batch_creator(batch_size, X_train.shape[0], 'train')
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                 y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_train,
                                      y: y_train}))
    
    print(tf.argmax(output_layer, 1).eval({x: X_test, y: y_test}))
    
    pred.append(tf.argmax(output_layer, 1).eval({x: X_test, y: y_test}))
    

