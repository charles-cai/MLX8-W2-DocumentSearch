#
#
#
import datasets
import hashlib
import pickle
import random


#
#
#
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1')


#
#
#
docs = {}
qrys = {}


#
#
#
for s in ds.keys():
  for e in ds[s]:
    qrys[e['query_id']] = { 'text': e['query'], 'docs': [] }
    for p in e['passages']['passage_text']:
      hsh = hashlib.sha256(p.encode()).hexdigest()[:16]
      if hsh not in docs: docs[hsh] = p
      qrys[e['query_id']]['docs'].append(hsh)


#
#
#
print("len(qrys)", len(qrys))
print("len(docs)", len(docs))
print(list(docs.keys())[:10])
print(list(qrys.keys())[:10])


#
#
#
print(docs['fdb37125d43984c2'])
print(random.choice(list(docs.values())))
print(qrys[9655])


#
#
#
with open('./corpus/docs.pkl', 'wb') as f: pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./corpus/qrys.pkl', 'wb') as f: pickle.dump(qrys, f, protocol=pickle.HIGHEST_PROTOCOL)