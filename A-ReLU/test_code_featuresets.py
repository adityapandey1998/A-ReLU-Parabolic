#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:19:24 2019

@author: adityapandey
"""


import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt

#import AReLU
import Leaky_AReLU as AReLU

#AReLU.set_kn(1.0, 1.0)
AReLU.set_kn(0.94, 1.1)
#AReLU.set_kn(1.06, 1.1)

seed = 128
rng = np.random.RandomState(seed)

phL_EC = pd.read_csv('pHL_EC.csv')
phL_EC.drop(['Index','P. Name'], axis=1, inplace=True)
phL_EC['P. Habitable Class'] = phL_EC['P. Habitable Class'].astype(str)
phL_EC = phL_EC.loc[phL_EC['P. Habitable Class'].isin(["non-habitable", "mesoplanet", "psychroplanet"])]



df_minority1 = phL_EC[phL_EC['P. Habitable Class']=='mesoplanet']
df_minority2 = phL_EC[phL_EC['P. Habitable Class']=='psychroplanet']
df_majority = phL_EC[phL_EC['P. Habitable Class']=='non-habitable']

df_majority_downsampled = resample(df_majority, 
                                 replace=False,     
                                 n_samples=500,    
                                 random_state=123)

df_minority1_upsampled = resample(df_minority1, 
                                 replace=True,     
                                 n_samples=70,    
                                 random_state=123)

df_minority2_upsampled = resample(df_minority2, 
                                 replace=True,     
                                 n_samples=48,    
                                 random_state=123)

#phL_EC = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled])

phL_EC = pd.concat([df_majority_downsampled, df_minority1_upsampled, df_minority2_upsampled])

star_feat = ['S. Hab Zone Min (AU)', 'S. Hab Zone Max (AU)', 'S. Luminosity (SU)', 'S. Mass (SU)', 'S. Radius (SU)', 'S. Teff (K)', 'P. Habitable Class']

set_1 = star_feat + ['P. Radius (EU)']
set_2 = star_feat + ['P. Mass (EU)']
set_3 = star_feat + ['P. Min Mass (EU)']

set_4 = ['P. Min Mass (EU)', 'P. Mass (EU)', 'P. Max Mass (EU)', 'P. Radius (EU)', 'P. Density (EU)', 'P. Gravity (EU)', 'S. Mass (SU)', 'S. Radius (SU)', 'S. Teff (K)', 'S. Luminosity (SU)',  'P. Habitable Class']

phL_EC_feat = phL_EC[set_2]

categories = phL_EC['P. Habitable Class']
#print(categories.value_counts())
num_cat = len(categories.value_counts())

values = np.array(categories)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
labels = pd.DataFrame(onehot_encoded)

#print(label_encoder.classes_)

data = phL_EC_feat.drop('P. Habitable Class', axis=1)

data.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=4, stratify=labels)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
#print(X_train.shape, y_train.shape,y_test.shape )


def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('X_'+dataset_name).iloc[batch_mask].values
    #print(batch_x)
    if dataset_name == 'train':
        batch_y = eval('y_'+dataset_name).iloc[batch_mask].values
        
    return batch_x, batch_y



# number of neurons in each layer
input_num_units = X_train.shape[1]
hidden_num_units1 = 5
hidden_num_units2 = 20
output_num_units = num_cat

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 500
batch_size = X_train.shape[0]
#batch_size = 512
learning_rate = 0.08

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units1, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

weights = {
    'hidden1': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_uniform([hidden_num_units1, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_uniform([hidden_num_units1], seed=seed)),
    'output': tf.Variable(tf.random_uniform([output_num_units], seed=seed))
}


hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer1 = AReLU.tf_ARelu(hidden_layer1)

output_layer = tf.add(tf.matmul(hidden_layer1, weights['output']), biases['output'])

logits = output_layer
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

display_step=10

pred = []
actual = []

loss_vals=[[],[]]

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, epochs+1):
        batch_x, batch_y = batch_creator(batch_size, X_train.shape[0], 'train')

        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:

            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                 y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
            loss_vals[0].append(step)
            loss_vals[1].append(loss)

    print("Optimization Finished!")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_train, y: y_train}))
    
    
    pred.append(tf.argmax(output_layer, 1).eval({x: X_test, y: y_test}))
    actual.append(tf.argmax(y, 1).eval({y: y_test}))
    
'''
AReLU.set_kn(0.54, 1.3)
AReLU.set_kn(0.83, 1.12)
loss_vals2=[[],[]]

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, epochs+1):
        batch_x, batch_y = batch_creator(batch_size, X_train.shape[0], 'train')

        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:

            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                 y: batch_y})
            #print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
            loss_vals2[0].append(step)
            loss_vals2[1].append(loss)

    print("Optimization Finished!")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_train, y: y_train}))
    
    
    pred.append(tf.argmax(output_layer, 1).eval({x: X_test, y: y_test}))
    actual.append(tf.argmax(y, 1).eval({y: y_test}))
    
'''
    
#pred = list(pred[0])
    
fig = plt.figure()
ax = plt.axes()

ax.plot(loss_vals[0], np.clip(loss_vals[1], 0, 1000))
#ax.plot(loss_vals2[0], np.clip(loss_vals2[1], 0, 1000))


cm = confusion_matrix(list(actual[0]), list(pred[0]), labels=[0, 1, 2])

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
    
conf_list = list(label_encoder.classes_)
tp,fp,tn,fn=perf_measure(cm, conf_list)
final_accuracy =0.0
final_precision = 0.0
final_recall = 0.0


for i in range(len(conf_list)):
    accuracy=(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) * 100
    precision = tp[i] / (tp[i]+fp[i]) * 100
    recall = tp[i] / (tp[i]+fn[i]) * 100
    Fscore = 2*(precision*recall)/(precision+recall)
    print("Precision of ",conf_list[i],": ",precision,sep="")
    print("Recall of ",conf_list[i],": ",recall,sep="")
    print("F-Score of ",conf_list[i],": ",Fscore,sep="")
    print("Accuracy w.r.t. ",conf_list[i],": ",accuracy,sep="")
    print()
    final_precision+=precision
    final_recall+=recall
    final_accuracy+=accuracy
    

cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
