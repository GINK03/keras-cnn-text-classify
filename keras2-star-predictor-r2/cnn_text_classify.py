import pickle
import gzip
import numpy as np
import random
import os
import sys
import statistics
import glob
import re
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Input, Activation, Dropout, Flatten, Dense, Reshape, merge
from keras.layers import Concatenate, Multiply, Conv1D, MaxPool1D, BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.core import Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K

def CBRD(inputs, filters=64, kernel_size=3, droprate=0.5):
  x = Conv1D(filters, kernel_size, padding='same',
            kernel_initializer='random_normal')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x


def DBRD(inputs, units=4096, droprate=0.35):
  x = Dense(units)(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(droprate)(x)
  return x

input_tensor = Input( shape=(100, 3793) )

x = Dense(3000, activation='relu')(input_tensor)
x = CBRD(x, 16)
x = CBRD(x, 16)
x = MaxPool1D()(x)

x = CBRD(x, 32)
x = CBRD(x, 32)
x = MaxPool1D()(x)

x = CBRD(x, 64)
x = CBRD(x, 64)
x = MaxPool1D()(x)

x = CBRD(x, 128)
x = CBRD(x, 128)
x = CBRD(x, 128)
x = MaxPool1D()(x)

x = CBRD(x, 128)
x = CBRD(x, 128)
x = CBRD(x, 128)
x = MaxPool1D()(x)

x = Flatten()(x)
x = Dense(1, activation='linear')(x)

model = Model(inputs=input_tensor, outputs=x)
model.compile(loss='mae', optimizer='sgd')

if '--train' in sys.argv:
  init = 0
  try:
    target_model = sorted(glob.glob('models/*.h5'))[-1]
    model.load_weights( target_model ) 
    init = int( re.search(r'/(\d{1,}).h5', target_model).group(1) )
    print('init state update', init)
  except Exception as e:
    print(e)
    ...
  for i in range(init, 1000):
    files = glob.glob('pairs/*.pkl.gz')
    ys, Xs = [], []
    for name in random.sample(files, 1000):
      try:
        y, X = pickle.loads( gzip.decompress( open(name, 'rb').read() ) )
      except EOFError as e:
        continue
      #print( X.shape )
      ys.append(y)
      Xs.append(X)
    ys,Xs = np.array(ys), np.array(Xs)
    model.fit(Xs, ys, epochs=10)
    model.save_weights('models/{:09d}.h5'.format(i))
