import os
import sys

import pickle
import gzip

import concurrent.futures 
def _map1(arr):
  try:
    i, lines = arr
    chars = set()
    for line in lines: 
      try:
        star, text = line.split(' __SEP__ ')
      except ValueError as e:
        continue
      for ch in list(text):
        chars.add(ch)
    return chars
  except Exception as e:
    print('Deep', e)

if '--make_char_index' in sys.argv:
  f = open('rakuten_reviews.txt')

  arrs = {}
  for fi, line in enumerate(f):
    if fi%10000 == 0:
      print('now iter', fi)

    line = line.strip()
    i = fi % 8
    if arrs.get(i) is None:
      arrs[i] = []
    arrs[i].append( line )

  arrs = [ (i, lines) for i, lines in arrs.items() ]

  chars = set()
  with concurrent.futures.ProcessPoolExecutor(max_workers=5) as exe:
    for _chars in exe.map(_map1, arrs):
      [chars.add(ch) for ch in _chars ]
  
  char_index = {}
  for index, ch in enumerate(list(chars)):
    char_index[ch] = index
  open('char_index.pkl.gz', 'wb').write( gzip.compress( pickle.dumps(char_index) ) )

if '--check_max_size' in sys.argv:
  f = open('rakuten_reviews.txt')
  maxs = []
  for fi, line in enumerate(f):
    if fi%10000 == 0:
      print('now iter', fi)
    line = line.strip()
    try:
      star, text = line.split(' __SEP__ ')
    except ValueError as e:
      continue
    maxs.append(len(text)) 
  width = max(maxs)
  open('width.pkl', 'wb').write( pickle.dumps(width) )

def _map2(arr):
  fi, line = arr
  try:
    star, text = line.split(' __SEP__ ')
  except ValueError as e:
    return 

  base = [ [0.0]*size for i in range(width) ]
  for index, ch in enumerate( list(text) ):
    print( char_index[ch] )
    base[index][ char_index[ch] ] = 1.0

  star = float(star)
  open('pairs/{:09d}.pkl.gz'.format(fi), 'wb').write( gzip.compress(pickle.dumps( (star, base) ) ) ) 

if '--make_pair' in sys.argv:
  char_index = pickle.loads( gzip.decompress(open('char_index.pkl.gz','rb').read() ) )
  size = len(char_index)
  width = pickle.loads( open('width.pkl', 'rb').read() ) 

  f = open('rakuten_reviews.txt')

  arrs = []
  for fi, line in enumerate(f):
    if fi%10000 == 0:
      print('now iter', fi)
    line = line.strip()
    arrs.append( (fi, line) ) 
  
  with concurrent.futures.ProcessPoolExecutor(max_workers=5) as exe:
    exe.map(_map2, arrs)
