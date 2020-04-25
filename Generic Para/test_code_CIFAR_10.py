#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adityapandey
"""

from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.layers import Layer
from keras import backend as K

def Abs_Func(x):
    return 0.65*K.pow(x,2) + 0.5*x

class Abs_Func_Class(Layer):

    def __init__(self, **kwargs):
        super(Abs_Func_Class, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Abs_Func_Class, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Abs_Func(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(Abs_Func_Class, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

batch_size = 256
num_classes = 10
epochs = 20
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

'''
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''

model.add(Flatten())
model.add(Dense(512))
#model.add(Activation('sigmoid'))
model.add(Activation(Abs_Func))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # Preprocessing and data augmentation:
    datagen = ImageDataGenerator(
        rotation_range=0,  
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,  
        zoom_range=0.,  
        channel_shift_range=0.,  
        fill_mode='nearest',
        cval=0.,  
        horizontal_flip=True,  
        vertical_flip=False,  
        rescale=None,
        validation_split=0.2)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        workers=4)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
