from __future__ import print_function
import sys
from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, ZeroPadding2D
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, load_model
from glob import glob
import pickle
import numpy as np

def build_model(sequence_length=None, filter_sizes=None, embedding_dim=None, vocabulary_size=None, num_filters=None, drop=None, idx_name=None):
  inputs = Input(shape=(sequence_length,), dtype='int32')
  embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
  reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

  conv_0   = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  pad_0    = ZeroPadding2D((1,1))(conv_0)
  conv_0_1 = Convolution2D(512, filter_sizes[0], 3, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(pad_0)

  conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  pad_1  = ZeroPadding2D((1,1))(conv_1)
  conv_1_1 = Convolution2D(512, filter_sizes[1], 3, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(pad_1)

  conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  pad_2  = ZeroPadding2D((1,1))(conv_1)
  conv_2_1 = Convolution2D(512, filter_sizes[2], 3, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(pad_1)
  conv_3 = Convolution2D(num_filters, filter_sizes[3], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  conv_4 = Convolution2D(num_filters, filter_sizes[4], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

  maxpool_0   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
  maxpool_0_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0_1)
  maxpool_1   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
  maxpool_1_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 0, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1_1)
  maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)
  maxpool_2_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2]  - 0, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2_1)
  maxpool_3 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[3] + -3, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)
  maxpool_4 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[4] + -2, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

  merged_tensor = merge([maxpool_0, maxpool_0_1, maxpool_1, maxpool_1_1, maxpool_2, maxpool_2_1, maxpool_3, maxpool_4], mode='concat', concat_axis=1)
  flatten = Flatten()(merged_tensor)
  dropout = Dropout(drop)(flatten)
  output = Dense(output_dim=len(idx_name), activation='softmax')(dropout)

  model = Model(input=inputs, output=output)
  adam = Adam()
  model.compile(optimizer=adam, loss='poisson', metrics=['accuracy'])

  return model

def init_train():
  print('Loading data')
  Xs = []
  Ys = []
  voc = {}
  maxlen = 0
  maxwords = 0
  buff = set()
  TARGET_DIR = 'data/*'
  idx_name = {}
  for i, filename in enumerate(glob(TARGET_DIR)):
    idx_name[i] = filename
    for line in open(filename).read().split('\n'):
       a = list(line)
       maxlen = max(maxlen, len(a))
       [buff.add(w) for w in a]
  maxwords = len(buff)
  voc[maxwords] = '___MAX___'
  voc['___META_MAXWORD___'] = maxwords
  voc['___META_MAXLEN___'] = maxlen
  print("maxwords %d"%maxwords)
  print("maxlen %d"%maxlen)
  print("idx name len %d"%len(idx_name))
  for i, filename in enumerate(glob(TARGET_DIR)):
    for line in set(filter(lambda x:x!='', open(filename).read().split('\n'))):
      X = [maxwords]*maxlen
      line = line.strip()
      for idx, ch in enumerate(list(line)):
        if voc.get(ch) == None:
          voc[ch] = len(voc)
        convert = voc[ch]
        X[idx] = convert
      Xs.append(X)
      y = [0.]*len(idx_name)
      y[i] = 1.
      Ys.append(y)
  X_train, X_test, y_train, y_test = train_test_split( Xs, Ys, test_size=0.1, random_state=42)
  open('vod.pkl', 'wb').write(pickle.dumps(voc))
  open('idx_name.pkl', 'wb').write(pickle.dumps(idx_name))
  sequence_length = maxlen
  vocabulary_size = maxwords
  embedding_dim   = 256*1
  filter_sizes    = [3,4,5,1,2]
  num_filters     = 512*1
  drop            = 0.5

  nb_epoch   = 10
  batch_size = 30
  return sequence_length, embedding_dim, filter_sizes, vocabulary_size, num_filters, drop, idx_name, \
  	X_train, X_test, y_train, y_test, batch_size, nb_epoch, Xs, Ys

def train():
  sequence_length, embedding_dim, filter_sizes, vocabulary_size, num_filters, drop, idx_name, \
  	X_train, X_test, y_train, y_test, batch_size, nb_epoch, Xs, Ys = init_train()
  model = build_model(sequence_length=sequence_length, \
  	filter_sizes=filter_sizes, \
	embedding_dim=embedding_dim, \
	vocabulary_size=vocabulary_size, \
	num_filters=num_filters, \
	drop=drop, \
	idx_name=idx_name)
  if '--all' in sys.argv:
    model.fit(Xs, Ys, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.save('cnn_text_clsfic_all.model')
  else:
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.save('cnn_text_clsfic.model')

def pred():
  voc = pickle.loads(open('vod.pkl', 'rb').read())
  maxlen = voc['___META_MAXLEN___']
  maxwords = voc['___META_MAXWORD___']
  idx_name = pickle.loads(open('idx_name.pkl', 'rb').read())
  model = load_model('cnn_text_clsfic.model')
  for line in sys.stdin:
    line = line.strip()
    buff = [maxwords]*maxlen
    for i, ch in enumerate(line):
      if voc.get(ch) is None:
        buff[i] = maxwords
      else:
        buff[i] = voc.get(ch)
    results = model.predict(np.array([buff]), verbose=0)
    preds = results#np.log(results) / 1.0
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    for result in results:
      logsoftmax = np.log(result)
      max_ent = list(sorted([(i,e) for i,e in enumerate(list(logsoftmax))], key=lambda x:x[1]*-1))[:10]
      _, mini = min(max_ent, key=lambda x:x[1])
      _, maxi = max(max_ent, key=lambda x:x[1])
      base = maxi - mini
      for ent in max_ent:
        id, prob = ent
        prob = (prob - mini)/base
        if int(float(prob)*100) == 0: 
          prob += .01
        #print(mini)
        #print(prob)
        print(idx_name.get(id).split('/').pop().split('.').pop(0), "%d"%(int(float(prob)*100)) + "%" )

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--pred' in sys.argv:
    pred()

