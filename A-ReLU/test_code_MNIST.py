#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:19:24 2019

@author: adityapandey
"""

import tensorflow as tf
from sklearn.metrics import confusion_matrix

#import AReLU
import Leaky_AReLU as AReLU
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

AReLU.set_kn(1.0, 1.0)
#AReLU.set_kn(0.94, 1.1)
#AReLU.set_kn(1.06, 1.1)

seed = 1

num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10

input_num_units = num_input
hidden_num_units1 = 256
hidden_num_units2 = 128
output_num_units = num_classes

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 400
#batch_size = X_train.shape[0]
batch_size = 256
learning_rate = 0.02

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
'''
weights = {
    'hidden1': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_uniform([hidden_num_units1, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_uniform([hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_uniform([output_num_units], seed=seed))
}
'''

hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer1 = AReLU.tf_ARelu(hidden_layer1)

hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])


output_layer = tf.add(tf.matmul(hidden_layer1, weights['output']), biases['output'])

logits = output_layer

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

display_step = 50

pred = []
actual = []


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
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
    t_loss, t_acc = sess.run([loss_op, accuracy], feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels})
    print("Testing Loss= " + \
          "{:.4f}".format(loss) + ", Testing Accuracy= " + \
          "{:.3f}".format(acc))
    
    pred.append(tf.argmax(output_layer, 1).eval({x: mnist.test.images,
                                      y: mnist.test.labels}))
    actual.append(tf.argmax(y, 1).eval({y: mnist.test.labels}))
    
cm = confusion_matrix(list(actual[0]), list(pred[0]), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
#confusionmatrix = np.matrix(cm)
def perf_measure(ans, conf_list):
    TP = []
    FP = []
    TN = []
    FN = []
    s=0
    total=0
    for i in range(len(conf_list)):
        for j in range(len(conf_list)):
            total+=ans[i][j]
    for i in range(len(conf_list)):
        TP.append(ans[i][i])
        s=0
        for j in range(len(conf_list)):
            if(j!=i):
                s+=ans[j][i]
        FP.append(s)
        FN.append(sum(ans[i])-TP[i])
        TN.append(total-TP[i]-FP[i]-FN[i])
    return(TP, FP, TN, FN)
    
conf_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tp,fp,tn,fn=perf_measure(cm, conf_list)
final_accuracy =0.0
final_precision = 0.0
final_recall = 0.0

'''
for i in range(len(conf_list)):
    accuracy=(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) * 100
    precision = tp[i] / (tp[i]+fp[i]) * 100
    recall = tp[i] / (tp[i]+fn[i]) * 100
    Fscore = 2*(precision*recall)/(precision+recall)
    print("Precision of ",conf_list[i],": ",round(precision,2),sep="")
    print("Recall of ",conf_list[i],": ",round(recall,2),sep="")
    print("F-Score of ",conf_list[i],": ",round(Fscore,2),sep="")
    print("Accuracy w.r.t. ",conf_list[i],": ",round(accuracy,2),sep="")
    print()
    final_precision+=precision
    final_recall+=recall
    final_accuracy+=accuracy

'''