#
#
#
import torch
import models
import pickle


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
qry = torch.tensor([words_to_ids[w] for w in 'what animal bark'.split(' ')]).to(dev)
doc = torch.stack([torch.tensor([words_to_ids[w] for w in x.split(' ')]) for x in ['dog cute', 'cat meows', 'computers fast']]).to(dev)


#
#
#
qry = two.qry(w2v.emb(qry).mean(dim=0))
doc = two.doc(w2v.emb(doc).mean(dim=1))


#
#
#
res = torch.nn.functional.cosine_similarity(qry, doc)
print(res)
