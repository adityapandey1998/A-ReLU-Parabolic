#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:44:22 2019

@author: adityapandey
"""

import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import itertools

#import AReLU
import Leaky_AReLU as AReLU

AReLU.set_kn(1.0, 1.0)
#AReLU.set_kn(0.84, 1.1)
#AReLU.set_kn(0.54, 1.3)
#print(AReLU.ARelu_k, AReLU.ARelu_n)


'''
Only +ve weights

FScore (UpSampled)
Leaky AReLU - 59.29, 98.45, nan   (0.54, 1.3)
            - 55.55, 98.68, 47.05 (0.84, 1.1)
            
Leaky ReLU  - 65.38, 98.76, 16.66

AReLU       - 64.70, 98.58, 50.00 (0.54, 1.3)
            - 55.55, 98.68, 47.05 (0.84, 1.1)
ReLU        - 65.38, 98.76, 16.66

FScore 
Leaky AReLU - nan, 98.48, nan    (0.54, 1.3)
            - 28.57, 98.77, 62.5 (0.84, 1.1)

Leaky ReLU  - 27.58, 98.25, nan


AReLU       - nan, 98.48, nan       (0.54, 1.3)
            - 27.272, 98.669, 44.44 (0.84, 1.1)

ReLU        - 30.77, 98.46, 28.57


'''

seed = 1
rng = np.random.RandomState(seed)

phL_EC = pd.read_csv('pHL_EC.csv')
phL_EC.drop(['Index','P. Name'], axis=1, inplace=True)
phL_EC['P. Habitable Class'] = phL_EC['P. Habitable Class'].astype(str)
#phL_EC = phL_EC.loc[phL_EC['P. Habitable Class'].isin(["non-habitable", "mesoplanet", "psychroplanet"])]

print(phL_EC['P. Habitable Class'].value_counts())


df_minority1 = phL_EC[phL_EC['P. Habitable Class']=='mesoplanet']
df_minority2 = phL_EC[phL_EC['P. Habitable Class']=='psychroplanet']
df_majority = phL_EC[phL_EC['P. Habitable Class']=='non-habitable']

phL_EC = pd.concat([df_minority1, df_minority2, df_majority])

df_majority_downsampled = resample(df_majority, 
                                 replace=True,     
                                 n_samples=1500,    
                                 random_state=123)

df_minority1_upsampled = resample(df_minority1, 
                                 replace=True,     
                                 n_samples=60,    
                                 random_state=123)

df_minority2_upsampled = resample(df_minority2, 
                                 replace=True,     
                                 n_samples=36,    
                                 random_state=123)

#phL_EC = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled])

#phL_EC = pd.concat([df_majority_downsampled, df_minority1_upsampled, df_minority2_upsampled])




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

#print(label_encoder.classes_)

data = phL_EC.drop('P. Habitable Class', axis=1)

data.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=4, stratify=labels)
batch_size =len(X_train)

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
hidden_num_units1 = 12
hidden_num_units2 = 20
output_num_units = num_cat

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 300
batch_size = X_train.shape[0]
batch_size = 512
learning_rate = 0.015

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


cm = confusion_matrix(list(actual[0]), list(pred[0]), labels=[i for i in range(num_cat)])

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
    print("Precision of ",conf_list[i],": ",round(precision, 2),sep="")
    print("Recall of ",conf_list[i],": ",round(recall, 2),sep="")
    print("F-Score of ",conf_list[i],": ",round(Fscore, 2),sep="")
    print("Accuracy w.r.t. ",conf_list[i],": ",round(accuracy, 2),sep="")
    print()
    final_precision+=precision
    final_recall+=recall
    final_accuracy+=accuracy
    

cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
#plot_confusion_matrix(cm2, conf_list)