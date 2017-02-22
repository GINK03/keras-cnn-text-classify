from __future__ import print_function
import sys
from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, ZeroPadding2D
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from data_helpers import load_data
from keras.optimizers import Adam
from keras.models import Model, load_model
from glob import glob



# this returns a compiled model
def build_model():
  inputs = Input(shape=(sequence_length,), dtype='int32')
  embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
  reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

  conv_0   = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  pad_0    = ZeroPadding2D((1,1))(conv_0)
  conv_0_1 = Convolution2D(512, filter_sizes[0], 3, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(pad_0)

  conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  pad_1  = ZeroPadding2D((1,1))(conv_1)
  conv_1_1 = Convolution2D(512, filter_sizes[0], 3, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(pad_1)

  conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
  conv_3 = Convolution2D(num_filters, filter_sizes[3], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

  maxpool_0   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
  maxpool_0_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0_1)
  maxpool_1   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
  maxpool_1_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1_1)
  maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)
  maxpool_3 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[3] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

  merged_tensor = merge([maxpool_0, maxpool_0_1, maxpool_1, maxpool_1_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
  flatten = Flatten()(merged_tensor)
  # reshape = Reshape((3*num_filters,))(merged_tensor)
  dropout = Dropout(drop)(flatten)
  output = Dense(output_dim=2, activation='softmax')(dropout)

  # this creates a model that includes
  model = Model(input=inputs, output=output)

  adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

print('Loading data')
#x, y, vocabulary, vocabulary_inv = load_data()
#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
Xs = []
Ys = []
voc = {}
maxlen = 0
maxwords = 0
buff = set()
TARGET_DIR = 'data.r/*'
idx_name = {}
for filename in glob(TARGET_DIR):
    for line in open(filename).read().split('\n'):
       a = list(line)
       maxlen = max(maxlen, len(a))
       [buff.add(w) for w in a]
maxwords = len(buff)
print("maxwords %d"%maxwords)
print("maxlen %d"%maxlen)
for i, filename in enumerate(glob(TARGET_DIR)):
  idx_name[i] = filename
  for line in open(filename).read().split('\n'):
    X = [maxwords]*maxlen
    line = line.strip()
    for idx, ch in enumerate(list(line)):
      if voc.get(ch) == None:
        voc[ch] = len(voc)
      convert = voc[ch]
      X[idx] = convert
    Xs.append(X)
    y = [0.]*2
    y[i] = 1.
    Ys.append(y)
X_train, X_test, y_train, y_test = train_test_split( Xs, Ys, test_size=0.2, random_state=42)
print("idx name len %d"%len(idx_name))
#print(y_train)
sequence_length = maxlen
vocabulary_size = maxwords
embedding_dim   = 256*1
filter_sizes    = [3,4,5,6]
num_filters     = 512*1
drop            = 0.5

nb_epoch = 30
batch_size = 30
def train():
  model = build_model()
  #checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))  # starts training
  model.save('cnn_text_clsfic.model')

def pred():
  model = load_model('cnn_text_clsfic.model')
  results = model.predict(X_test, verbose=1)
  for result in results:
    #max_ent = max([(i,e) for i,e in enumerate(list(result))], key=lambda x:x[1])
    max_ent = [(i,e) for i,e in enumerate(list(result))]
    print(max_ent)
  pass
if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--pred' in sys.argv:
    pred()

