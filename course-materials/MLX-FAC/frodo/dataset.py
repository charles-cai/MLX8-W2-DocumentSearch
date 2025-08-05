#
#
#
#
import torch
import subprocess
import more_itertools
import random
import models
import pandas
import pickle
import time


#
#
#
#
class Window(torch.utils.data.Dataset):
  def __init__(self, path, size=5):
    self.path = path
    self.size = size
    with open(self.path, 'r') as f: self.tkns = list(map(int, f.read().strip().split()))
    self.wins = list(more_itertools.windowed(self.tkns, n=size))
    m = (self.size - 1) // 2
    self.inps = [w[m] for w in self.wins]
    self.trgs = [[w[m - 1], w[m + 1]] for w in self.wins]

  def __getitem__(self, idx):
    return torch.tensor(self.inps[idx]), torch.tensor(self.trgs[idx])

  def __len__(self):
    return len(self.inps)


#
#
#
#
class Triplets(torch.utils.data.Dataset):
  def __init__(self, embs, tkns):
    self.embs = embs
    self.tkns = tkns
    self.qrys = pickle.load(open('./corpus/qrys.pkl', 'rb'))
    self.docs = pickle.load(open('./corpus/docs.pkl', 'rb'))
    self.q_keys = list(self.qrys.keys())
    self.d_keys = list(self.docs.keys())
    with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
    self.words_to_ids = tkns['words_to_ids']

  def __len__(self):
    return len(self.qrys)

  def __getitem__(self, idx):
    qry = self.qrys[self.q_keys[idx]]
    pos = self.docs[qry['docs'][0]]
    neg = self.docs[random.choice(self.d_keys)]
    qry = self.to_emb(qry['text'])
    pos = self.to_emb(pos)
    neg = self.to_emb(neg)
    return qry, pos, neg

  def to_emb(self, text):
    text = self.preprocess(text)
    tkns = [self.tkns[t] for t in text if t in self.tkns]
    if len(tkns) == 0: return
    tkns = torch.tensor(tkns).to('cuda:0')
    embs = self.embs(tkns)
    return embs.mean(dim=0)

  def preprocess(self, text):
    text = text.lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace('“',  ' <QUOTATION_MARK> ')
    text = text.replace('”',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    text = text.replace("'",  ' <APOSTROPHE> ')
    text = text.replace("’",  ' <APOSTROPHE> ')
    return text.split()


#
#
#
#
if __name__ == '__main__':
  # import tqdm
  # ds = StreamingWords('./corpus/38k5fm.txt')
  # dl = torch.utils.data.DataLoader(ds, batch_size=2)
  # for idx, elm in enumerate(tqdm.tqdm(ds)): print("ds", elm); time.sleep(1)
  # for idx, elm in enumerate(tqdm.tqdm(dl)): print("dl", elm); time.sleep(1)

  # with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
  # words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']

  # w2v = models.SkipGram(voc=len(words_to_ids), emb=128)
  # w2v.load_state_dict(torch.load('./checkpoints/3.xjrg.w2v.pth'))
  # ds = TripletsDataset('./corpus/triples.csv', w2v)
  # q, p, n = ds[0]; print(q.shape, p.shape, n.shape)
  pass