#
#
#
import torch
import models
import pickle
import dataset
import fastapi


#
#
#
with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#
w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev)
w2v.load_state_dict(torch.load('./checkpoints/2025_02_06__18_31_03.0.70000.w2v.pth'))


#
#
#
two = models.Towers(emb=128).to(dev)
two.load_state_dict(torch.load('./checkpoints/2025_02_06__19_08_18.0.350.two.pth'))


#
#
#
ds = dataset.Triplets(w2v.emb, words_to_ids)
db = torch.stack([two.doc(ds.to_emb(ds.docs[k]).to(dev)) for k in ds.d_keys])


#
#
#
app = fastapi.FastAPI()


#
#
#
@app.get("/search")
async def search(q: str):
  if q.strip() == '': return []
  qry = ds.to_emb(q)
  if qry is None: return []
  qry = two.qry(qry)
  res = torch.nn.functional.cosine_similarity(qry, db)
  top_scr, top_idx = torch.topk(res, k=4)
  return [{'score': s.item(), 'doc': ds.docs[ds.d_keys[i.item()]]} for s, i in zip(top_scr, top_idx)]
