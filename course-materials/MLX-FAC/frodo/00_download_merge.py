#
#
#
#
import datasets
import requests


#
#
#
#
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1')
docs = [passage for s in ds.keys() for e in ds[s] for passage in e['passages']['passage_text']]
qrys = [e['query'] for s in ds.keys() for e in ds[s]]
with open('./corpus/msmarco.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(docs + qrys)))


#
#
#
#
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('./corpus/text8.txt', 'wb') as f: f.write(r.content)
