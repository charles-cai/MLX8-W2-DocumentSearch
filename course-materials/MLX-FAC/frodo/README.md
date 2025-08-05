## Frodo


```sh
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init --all
```

```sh
$ conda create --name frd python=3.11 -y
$ conda activate frd
$ pip install datasets torch more-itertools wandb
```


```sh
$ python 00_download_merge.py
$ python 01_tokenise_corpus.py
$ python 02_doc_qry_data.py
$ python 03_train.w2v.py
$ python 04_train.two.py
$ nohup uvicorn 05_server:app --host 0.0.0.0 --port 8000 > _frodo.log 2>&1 &
```