#
#
#
import torch
import pickle
import models


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
qry = w2v.emb(torch.tensor(words_to_ids['computer']).to(dev)).squeeze()
res = torch.nn.functional.cosine_similarity(qry, w2v.emb.weight)
top_v, top_i = torch.topk(res, k=4)
print([ids_to_words[i.item()] for i in top_i])
