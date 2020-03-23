#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:36:24 2020

@author: adityapandey
"""

import numpy as np

import tensorflow as tf

import Para_Func
from keras.datasets import cifar10

#(trainX, trainy), (testX, testy) = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))


seed = 1
rng = np.random.RandomState(seed)


X_train_norm = X_train.astype('float32')
X_test_norm = X_test.astype('float32')
X_train_norm = X_train_norm / 255.0
X_test_norm = X_test_norm / 255.0

# set remaining variables
epochs = 300
batch_size = X_train.shape[0]
batch_size = 512
learning_rate = 0.015



init = tf.global_variables_initializer()

display_step=10

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  print(features)
  
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
  print(input_layer)
  
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print(pool1)
  
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same") #to try other activation function replcae relu6 by leaky_relu or relu
  
  conv2_act = Para_Func.tf_para_func(conv2)
  
  pool2 = tf.layers.max_pooling2d(inputs=conv2_act, pool_size=[2, 2], strides=2)
  '''
  
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) #to try other activation function replcae relu6 by leaky_relu or relu
  
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=10)
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),

      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
   }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  '''
  
  
# Test
  
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train_norm},
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

#mnist_classifier.train(input_fn=train_input_fn, steps=1,hooks=[logging_hook])
mnist_classifier.train(input_fn=train_input_fn, steps=1000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test_norm},
    y=y_test,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

print(eval_results)