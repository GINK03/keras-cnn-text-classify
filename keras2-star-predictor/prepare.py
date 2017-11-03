import json
import MeCab
import sys
import numpy as np
import pickle
from scipy import sparse
import glob
from scipy.sparse import csr_matrix
import gzip
if '--shrinkage' in sys.argv:
  m  = MeCab.Tagger('-Owakati')
  for line in open('reveiws.json'):
    line = line.strip()
    obj = json.loads(line)
    title = obj['reviewTitle'].strip()
    review = obj['review'].strip()
    star = obj['stars']
    if 'ネタバレ' in review:
      continue
    if len(title) > 50:
      continue
    if len(review) > 500:
      continue
    print( json.dumps( (star, title, review), ensure_ascii=False )  )

if '--to_index' in sys.argv:
  chars = set()
  for line in open('shrinkage.json'):
    line = line.strip()
    star, title, review = json.loads(line)
    [chars.add(char) for char in list(title)]
    [chars.add(char) for char in list(review)]

  char_index, index_char = {}, {}
  for index, char in enumerate(chars):
    print(char, index)
    char_index[char] = index
    index_char[index] = char

  open('char_index.json','w').write( json.dumps(char_index, indent=2, ensure_ascii=False) )
  open('index_char.json','w').write( json.dumps(index_char, indent=2, ensure_ascii=False) )

if '--to_array' in sys.argv:
  char_index = json.loads(open('char_index.json','r').read() )
  index_char = json.loads(open('index_char.json','r').read() )

  for index, line in enumerate(open('shrinkage.json')):
    print('now iter', index)
    line = line.strip()
    star, ts, rs = json.loads(line)
    ts, rs = list(ts), list(rs)
    tv = [0.0]*50
    rv = [0.0]*500
    for i,t in enumerate(ts):
      tv[i] = char_index[t]
    for i,r in enumerate(rs):
      #print(i, r)
      rv[i] = char_index[r]
   
    tv = np.array(tv, dtype=np.int)
    rv = np.array(rv, dtype=np.int)
    open('dataset/{index}.pkl'.format(index=index), 'wb').write( gzip.compress( pickle.dumps( ( star, tv, rv) )  ) )

