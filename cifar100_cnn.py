# -*- coding: utf-8 -*-
# cifar100_cnn.py

""" 
  Train a simple CNN on Cifar100 small image datasets.
  
  The architecture is taken from official site of keras examples.
    https://github.com/keras-team/keras/tree/master/examples
  
  Usage:
    'python cifar100_cnn.py <label_mode>'
    
    where <label_mode> = 'fine' or 'coarse'
    
"""

import numpy as np
import utils
import os
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

def _print_usage():
  """ Prints how to pass arguments on the command line

  Args:
    None

  Returns:
    None
  """
  print('Usage:')
  print('python cifar100_cnn.py <label_mode>')
  print("where <label_mode> = 'fine' or 'coarse'")
  
if len(sys.argv) != 2:
  _print_usage()
  raise SyntaxError('label_mode missing in command')
  
label_mode = sys.argv[1]
print("Label mode: {}".format(label_mode))

if label_mode not in ['fine', 'coarse']:
  _print_usage()
  raise SyntaxError('Incorrect label_mode passed')

# Set the hyperparameters
batch_size = 32
n_epochs = 100

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar100_{}_cnn_model.h5'.format(label_mode)  
  
# Load Cifar100 dataset
x, y_c, y_f, xt, yt_c, yt_f = utils.load_dataset()

X_train = x.astype('float32')
X_test = xt.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if label_mode == 'fine':
  n_classes = 100
  y_train = y_f
  y_test = yt_f
else:
  n_classes = 20
  y_train = y_c
  y_test = yt_c

# Convert to one hot encoding
Y_train = np.eye(n_classes)[y_train]
Y_test = np.eye(n_classes)[y_test]

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# Optimizer
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rescale = 1./255,  
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    horizontal_flip=True)  

datagen.fit(X_train)

X_test = X_test/255.

# Train
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)//batch_size + 1,
                    epochs=n_epochs,
                    validation_data=(X_test, Y_test),
                    verbose=2)  

# Save model                    
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Evaluate performance
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])              