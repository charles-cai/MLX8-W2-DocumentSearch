## DuckDB Installation

https://duckdb.org/#quickinstall

```shell
curl https://install.duckdb.org | sh
```

## Tests Connecting to DuckDB and Querying Hugging Face Datasets

DuckDB accessing Hugging Face Datasets: https://duckdb.org/2024/05/29/access-150k-plus-datasets-from-hugging-face-with-duckdb.html

Hugging Face Documentation: https://huggingface.co/docs/hub/en/datasets-duckdb

```sql
SELECT * FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 5;

SELECT * FROM 'hf://datasets/microsoft/ms_marco/v1.1/train-00000-of-00001.parquet' LIMIT 5;

SELECT * FROM 'hf://datasets/microsoft/ms_marco/v1.1/validation-00000-of-00001.parquet' LIMIT 5;
```

## DuckDB JSON Extraction Functions:

https://duckdb.org/docs/stable/data/json/json_functions

```sql

SELECT json_extract(passages, '$is_selected') FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 1;

--- {
--    'is_selected': 
--      [0, 0, 1, 0, 0, 0, 0], 
--    'passage_text': 
--      [We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing f… 
--  }
SELECT json_extract(passages, '$.is_selected') FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 1;
-- [0,0,1,0,0,0,0]   

SELECT json_extract(passages, '$.passage_text') FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 1;
-- ["We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing fur. One has a patch missing down his back and under bo…  │

SELECT json_extract(passages, '$.passage_text[1]') FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 1;
--  "Critters cannot stand the smell of human hair, so sprinkling a barrier of hair clippings around your garden, or lightly working it into the soil when you plant bulbs, appar…  │

SELECT passages -> '$.is_selected' FROM 'hf://datasets/microsoft/ms_marco/v1.1/test-00000-of-00001.parquet' LIMIT 2;
-- 
-- [0,0,1,0,0,0,0]               │
-- [0,1,0,0,0,0,0,0,0]
  
```

## Local Parquet files

Raw Hugging Face MS MARCO data dets:
```sql
SELECT * FROM '.data/ms_marco_v1.1/train.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/ms_marco_v1.1/train.parquet' LIMIT 10;
DESCRIBE '.data/ms_marco_v1.1/train.parquet';

SELECT * FROM '.data/ms_marco_v1.1/validation.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/ms_marco_v1.1/validation.parquet' LIMIT 10;
DESCRIBE '.data/ms_marco_v1.1/validation.parquet';

SELECT * FROM '.data/ms_marco_v1.1/test.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/ms_marco_v1.1/test.parquet' LIMIT 10;
DESCRIBE '.data/ms_marco_v1.1/test.parquet';

-- JSON 
SELECT passages -> '$.is_selected' FROM '.data/ms_marco_v1.1/train.parquet' LIMIT 2;
SELECT MAX(query_id) as max, MIN(query_id) as min, COUNT (*), max-min + 1 as diff FROM  '.data/ms_marco_v1.1/train.parquet';
SELECT passages -> '$.passage_text' FROM '.data/ms_marco_v1.1/test.parquet' LIMIT 2;
SELECT passages -> '$.is_selected' AS s, passages -> '$.url' AS url FROM '.data/ms_marco_v1.1/test.parquet' LIMIT 2;
SELECT passages -> '$.passage_text' FROM '.data/ms_marco_v1.1/test.parquet' LIMIT 1;

-- ["We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing fur. One has a patch miss…  ]

-- JSON Table functions: https://duckdb.org/docs/stable/data/json/json_functions#json-table-functions

SELECT p.* FROM '.data/ms_marco_v1.1/test.parquet' AS test, json_each(test.passages) AS p LIMIT 2;
--
-- ┌──────────────┬───────────────────────────────────────────┬─────────┬──────┬────────┬────────┬────────────────┬─────────┐
-- │     key      │      value                                │  type   │ atom │   id   │ parent │    fullkey     │  path   │
-- │   varchar    │      json                                 │ varchar │ json │ uint64 │ uint64 │    varchar     │ varchar │
-- ├──────────────┼───────────────────────────────────────────┼─────────┼──────┼────────┼────────┼────────────────┼─────────┤
-- │ is_selected  │ [0,0,0,0,0,1,0,0,0]                       │ ARRAY   │ NULL │      2 │   NULL │ $.is_selected  │ $       │
-- │ passage_text │ ["Congratulations! You have found, … "    │ ARRAY   │ NULL │     13 │   NULL │ $.passage_text │ $       │
-- └──────────────┴───────────────────────────────────────────┴─────────┴──────┴────────┴────────┴────────────────┴─────────┘

```

Processed Triples Dataset
```sql
SELECT * FROM '.data/processed/train_triples.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/train_triples.parquet' LIMIT 10;
DESCRIBE  '.data/processed/train_triples.parquet';

SELECT * FROM '.data/processed/validation_triples.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/validation_triples.parquet' LIMIT 10;
DESCRIBE  '.data/processed/validation_triples.parquet';

SELECT * FROM '.data/processed/test_triples.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/test_triples.parquet' LIMIT 10;
DESCRIBE  '.data/processed/test_triples.parquet'; 


--
-- ┌────────┬───────┬──────────────┬───────┐
-- │  max   │  min  │ count_star() │ diff  │
-- │ int32  │ int32 │    int64     │ int32 │
-- ├────────┼───────┼──────────────┼───────┤
-- │ 102128 │ 19699 │    82326     │ 82430 │
-- └────────┴───────┴──────────────┴───────┘
--
```

With word2vec embeddings precalculated:
```sql
-- train
SELECT * FROM '.data/processed/train_triples_embeddings.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/train_triples_embeddings.parquet' LIMIT 10;
DESCRIBE '.data/processed/train_triples_embeddings.parquet';

-- validation
SELECT * FROM '.data/processed/validation_triples_embeddings.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/validation_triples_embeddings.parquet' LIMIT 10;
DESCRIBE '.data/processed/validation_triples_embeddings.parquet';

-- test
SELECT * FROM '.data/processed/test_triples_embeddings.parquet' LIMIT 10;
SELECT COUNT(*) FROM '.data/processed/test_triples_embeddings.parquet' LIMIT 10;
DESCRIBE '.data/processed/test_triples_embeddings.parquet';
```