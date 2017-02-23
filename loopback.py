from twitter import *
import json
import os
import math
import plyvel 
import random
import sys
import subprocess

SELF_NAME = "@deep_shigure"
RES_REPO = ["うーん、たぶんね。。。", \
	"ちょっと難しいけど思い出してみたよ", \
	"ふふ、難しいことを聞くね" \
	"きっと彼女たちかな", \
	"僕は忘れないよ"]
def dump():
  db = plyvel.DB('log.ldb', create_if_missing=False) 
  for k, v in db:
    k,v = (k.decode('utf-8'), v.decode('utf-8'))
    print(v)
    obj = json.loads(v)
    if obj.get('user') == None:
      continue
    screen_name = obj['user']['screen_name']
    text        = obj['text']
    print(screen_name, text)

def create_rep(screen_name, text):
  reptag = "@%s"%screen_name
  text   = text.replace("%s "%SELF_NAME, '')
  command = "echo %s | python3 model.py --pred"%text
  proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  response, _ = proc.communicate()
  print("response \n", response.decode('utf-8'))
  print("err \n ", _)
  random.shuffle(RES_REPO)
  rep = "%s\n%s\n%s"%(reptag, RES_REPO[0], response.decode('utf-8'))
  if len(rep) > 140:
    rep = rep[:140]
  return rep

def main():
  db = plyvel.DB('log.ldb', create_if_missing=True) 
  ckey, csec, atkn, asec, _ = open('keys.txt').read().split('\n')
  print(ckey, csec, atkn, asec, _ )
  auth = OAuth(atkn, asec, ckey, csec)
  t = Twitter(auth=auth)
  #tls = t.statuses.home_timeline()
  #for tl in tls:
  #  print(tl)
  t_userstream = TwitterStream(auth=auth,domain='userstream.twitter.com')
  for i, msg in enumerate(t_userstream.user()):
    hash = "%032x"%random.getrandbits(128)
    log = json.dumps(msg, indent=4)
    #print(log)
    db.put(bytes(hash, 'utf-8'), bytes(log, 'utf-8'))
    obj = msg
    if obj.get('user') == None:
      continue
    screen_name = obj['user']['screen_name']
    print("screen_name", screen_name)
    if SELF_NAME.replace('@', '') in screen_name:
      print("自己参照です")
      continue
    text        = obj['text']
    rep = create_rep(screen_name, text)
    print("generate ress", rep)
    t.statuses.update(status=rep)
if __name__ == '__main__':
  if '--test_rep' in sys.argv:
    rep = create_rep("nardtree", "はわわ～ そろそろなのです")
    print(rep, len(rep))
  if '--run' in sys.argv:
    main()
  if '--dump' in sys.argv:
    dump()
