from __future__ import print_function
import sys
if '--plaid' in sys.argv:
  # Install the plaidml backend
  import plaidml.keras
  plaidml.keras.install_backend()

from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Concatenate, multiply, Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.regularizers    import l2
from keras.layers.core     import Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization as BN
from keras.applications.vgg16 import VGG16 
from scipy.sparse import csr_matrix  
import keras.backend as K
import numpy as np
import random
import pickle
import glob
import copy
import os
import re
import json
import gzip

edim      = 256

inputs    = Input(shape=(500,), dtype='int32')
x         = Embedding(output_dim=edim, input_dim=5000, input_length=500)(inputs)
x         = Reshape( (500,edim,1) )(x)

maxpools = []
for i in [2,3,4,5]:
  conv_0    = Convolution2D(512, (i, edim), padding="valid", activation="relu", data_format="channels_last", kernel_initializer="normal")(x)
  pad_0     = ZeroPadding2D((1,1))(conv_0)
  maxpool_0 = MaxPooling2D(pool_size=(500 - i + 1, 1), strides=(1,1),  padding="valid", data_format="channels_last")(conv_0)
  maxpools.append(maxpool_0)

x         = Concatenate(axis=-1)(maxpools)
x         = Dropout(0.35)(x)
x         = Flatten()(x)
x         = Dense(units=1, activation='linear')(x)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

def train():
  ys,Xs = [],[]
  for index, name in enumerate(glob.glob('dataset/*.pkl')):
    y,_,x = pickle.loads( gzip.decompress( open(name, 'rb').read() ) )
    ys.append(y)
    Xs.append(x)
    if index%1000 == 0:
      print(index, x.shape)
  ys = np.array(ys)
  Xs = np.array(Xs)
  print(ys.shape)
  print(Xs.shape)
  for i in range(10):
    model.fit(Xs,ys,epochs=1)
    model.save_weights('models/{:09d}.h5'.format(i))

if '--train' in sys.argv:
  train()
