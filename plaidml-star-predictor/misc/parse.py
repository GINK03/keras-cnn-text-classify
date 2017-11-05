import json

import sys

if '--char' in sys.argv:
  for line in open('./reviews.json'):
    line = line.strip()
    o = json.loads(line)
    rv = o['review']
    print( ' '.join( list(rv) ) )
      
    #star = o['stars']
import pickle
if '--pickle' in sys.argv:
  char_vec = {}
  for line in open('./model.vec'):
    line = line[:-1]
    es = line.split()
    ch = es.pop(0)
    char_vec[ch] = [float(e) for e in es]

    print(ch)
  open('char_vec.pkl','wb').write( pickle.dumps(char_vec) )

if '--data_gen' in sys.argv:
  char_vec = pickle.loads( open('char_vec.pkl','rb').read() )
  ys = []
  Xs = []
  for oindex, line in enumerate(open('./reviews.json')):
    if oindex  > 10000:
      break
    line = line.strip()
    o = json.loads(line)
    rv = o['review']
    stars = o['stars'] 
    if stars is None:
      continue
    X = [ [ 0.0 for i in range(256) ] for j in range(256) ]
    for index, ch in enumerate( list(rv)[:256]) :
      if char_vec.get(ch) is None:
        continue
      #print(char_vec[ch])
      X[index] = char_vec[ch]
    y = float(stars)
    ys.append(y)
    Xs.append(X)
  open('dataset.pkl', 'wb').write( pickle.dumps( (ys,Xs) ) )
