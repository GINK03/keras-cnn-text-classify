import os
import sys

import pickle
import gzip
import random
import concurrent.futures 
import json
import zipfile
import numpy as np
if '--tag_index' in sys.argv:
  tag_freq = {}
  z = zipfile.ZipFile('tag_introduction.zip')

  for name in z.namelist():
    f = z.open(name)
    raw = f.read().decode('utf8') 
    try:
      obj = json.loads(raw)
    except:
      continue
    for tag in obj['tags']:
      if tag_freq.get(tag) is None:
        tag_freq[tag] = 0
      tag_freq[tag] += 1

  tag_index = {} 
  for index, (tag, freq) in enumerate( random.sample(sorted(tag_freq.items(), key=lambda x:x[1]*-1)[:10000], k=10000)):
    print( index, tag, freq )
    tag_index[tag] = index
  open('tag_index.json', 'w').write( json.dumps(tag_index, indent=2, ensure_ascii=False) )

if '--char_index' in sys.argv:
  tag_freq = {}
  z = zipfile.ZipFile('tag_introduction.zip')
  chars = set()
  for name in z.namelist():
    f = z.open(name)
    raw = f.read().decode('utf8') 
    try:
      obj = json.loads(raw)
    except:
      continue
    for char in list(obj['intro']):
      chars.add(char)

  char_index = {char:index for index, char in enumerate(chars)}
  open('char_index.json', 'w').write( json.dumps(char_index, indent=2, ensure_ascii=False) )

def _map2(arr):
  fi, line = arr

  if os.path.exists('pairs/{:09d}.pkl.gz'.format(fi)) is True:
    return "Already processed."
  try:
    star, text = line.split(' __SEP__ ')
  except ValueError as e:
    return 
  
  base = [ [0.0]*size for i in range(width) ]
  for index, ch in enumerate( list(text) ):
    #print( char_index[ch] )
    try:
      base[index][ char_index[ch] ] = 1.0
    except Exception as e:
      break

  star = float(star)
  base = np.array(base)
  print('now iter', fi)
  open('pairs/{:09d}.pkl.gz'.format(fi), 'wb').write( gzip.compress(pickle.dumps( (star, base) ) ) ) 

if '--make_pair' in sys.argv:
  char_index = json.loads( open('char_index.json').read()  )
  tag_index = json.loads( open('tag_index.json').read()  )
  char_size = len(char_index)
  tag_size = len(tag_index)
  
  z = zipfile.ZipFile('tag_introduction.zip')
  chars = set()
  for name in z.namelist():
    
    f = z.open(name)
    raw = f.read().decode('utf8') 
    save_name = 'pairs/{}.pkl.gz'.format(name.split('/').pop())
    if os.path.exists(save_name) is True:
      continue
    try:
      obj = json.loads(raw)
    except:
      continue
    y = [tag_index[tag] for tag in obj['tags'] if tag_index.get(tag) is not None]
    X = [char_index[char] for char in obj['intro'] ]
    
    baseX = [ [0.0]*char_size for i in range(200) ]
    for index, Xi in enumerate(X[:200]):
      baseX[index][Xi] = 1.0

    baseY = [0.0]*tag_size  
    for index, Yi in enumerate(y[:50]):
      baseY[Yi] = 1.0 

    try:
      open(save_name, 'wb').write( gzip.compress(pickle.dumps( ( name, baseX, baseY ) ) ) )
    except OSError as ex:
      continue
    print( name )
