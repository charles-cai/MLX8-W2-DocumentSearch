#
#
#
import torch


#
#
#
#
class SkipGram(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs, rand):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt


#
#
#
#
class QryTower(torch.nn.Module):
  def __init__(self, emb):
    super().__init__()
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=emb, out_features=emb),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=emb, out_features=emb)
    );

  def forward(self, avg):
    out = self.mlp(avg)
    return out


#
#
#
#
class DocTower(torch.nn.Module):
  def __init__(self, emb):
    super().__init__()
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=emb, out_features=emb),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=emb, out_features=emb)
    );

  def forward(self, avg):
    out = self.mlp(avg)
    return out


#
#
#
#
class Towers(torch.nn.Module):
  def __init__(self, emb):
    super().__init__()
    self.qry = QryTower(emb)
    self.doc = DocTower(emb)

  def forward(self, qry, pos, neg, mrg):
    qry = self.qry(qry)
    pos = self.doc(pos)
    neg = self.doc(neg)
    pos = 1 - torch.nn.functional.cosine_similarity(qry, pos)
    neg = 1 - torch.nn.functional.cosine_similarity(qry, neg)
    return torch.max(pos - neg + mrg, torch.tensor(0.0)).mean()


#
#
#
#
if __name__ == '__main__':
  # towers = Towers(emb=128, mrg=0.1)
  # qry = torch.randn(1, 128)
  # pos = torch.randn(5, 128)
  # neg = torch.randn(5, 128)
  # print(towers(qry, pos, neg))
  pass
