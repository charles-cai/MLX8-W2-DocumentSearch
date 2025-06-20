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
    
-- String Length Analysis for Train Dataset
SELECT 
    'train' as dataset,
    AVG(LENGTH(query)) as avg_query_length,
    MIN(LENGTH(query)) as min_query_length,
    MAX(LENGTH(query)) as max_query_length,
    AVG(LENGTH(positive_doc)) as avg_positive_doc_length,
    MIN(LENGTH(positive_doc)) as min_positive_doc_length,
    MAX(LENGTH(positive_doc)) as max_positive_doc_length,
    AVG(LENGTH(negative_doc)) as avg_negative_doc_length,
    MIN(LENGTH(negative_doc)) as min_negative_doc_length,
    MAX(LENGTH(negative_doc)) as max_negative_doc_length
FROM '.data/processed/train_triples.parquet';

-- String Length Analysis for Validation Dataset
SELECT 
    'validation' as dataset,
    AVG(LENGTH(query)) as avg_query_length,
    MIN(LENGTH(query)) as min_query_length,
    MAX(LENGTH(query)) as max_query_length,
    AVG(LENGTH(positive_doc)) as avg_positive_doc_length,
    MIN(LENGTH(positive_doc)) as min_positive_doc_length,
    MAX(LENGTH(positive_doc)) as max_positive_doc_length,
    AVG(LENGTH(negative_doc)) as avg_negative_doc_length,
    MIN(LENGTH(negative_doc)) as min_negative_doc_length,
    MAX(LENGTH(negative_doc)) as max_negative_doc_length
FROM '.data/processed/validation_triples.parquet';

-- String Length Analysis for Test Dataset
SELECT 
    'test' as dataset,
    AVG(LENGTH(query)) as avg_query_length,
    MIN(LENGTH(query)) as min_query_length,
    MAX(LENGTH(query)) as max_query_length,
    AVG(LENGTH(positive_doc)) as avg_positive_doc_length,
    MIN(LENGTH(positive_doc)) as min_positive_doc_length,
    MAX(LENGTH(positive_doc)) as max_positive_doc_length,
    AVG(LENGTH(negative_doc)) as avg_negative_doc_length,
    MIN(LENGTH(negative_doc)) as min_negative_doc_length,
    MAX(LENGTH(negative_doc)) as max_negative_doc_length
FROM '.data/processed/test_triples.parquet';

-- Word Count Analysis (approximate using space splits)
SELECT 
    'train' as dataset,
    AVG(LENGTH(query) - LENGTH(REPLACE(query, ' ', '')) + 1) as avg_query_words,
    AVG(LENGTH(positive_doc) - LENGTH(REPLACE(positive_doc, ' ', '')) + 1) as avg_positive_doc_words,
    AVG(LENGTH(negative_doc) - LENGTH(REPLACE(negative_doc, ' ', '')) + 1) as avg_negative_doc_words
FROM '.data/processed/train_triples.parquet';

-- Distribution of string lengths (buckets)
SELECT 
    CASE 
        WHEN LENGTH(query) <= 50 THEN '0-50'
        WHEN LENGTH(query) <= 100 THEN '51-100'
        WHEN LENGTH(query) <= 200 THEN '101-200'
        WHEN LENGTH(query) <= 500 THEN '201-500'
        ELSE '500+' 
    END as query_length_bucket,
    COUNT(*) as count
FROM '.data/processed/train_triples.parquet'
GROUP BY query_length_bucket
ORDER BY query_length_bucket;

-- Document length distribution
SELECT 
    CASE 
        WHEN LENGTH(positive_doc) <= 500 THEN '0-500'
        WHEN LENGTH(positive_doc) <= 1000 THEN '501-1000'
        WHEN LENGTH(positive_doc) <= 2000 THEN '1001-2000'
        WHEN LENGTH(positive_doc) <= 5000 THEN '2001-5000'
        ELSE '5000+' 
    END as doc_length_bucket,
    COUNT(*) as count
FROM '.data/processed/train_triples.parquet'
GROUP BY doc_length_bucket
ORDER BY doc_length_bucket;

-- Sample long and short texts for inspection
SELECT 'SHORT_QUERY' as type, query, LENGTH(query) as length 
FROM '.data/processed/train_triples.parquet' 
WHERE LENGTH(query) < 20 
LIMIT 5;

SELECT 'LONG_QUERY' as type, query, LENGTH(query) as length 
FROM '.data/processed/train_triples.parquet' 
WHERE LENGTH(query) > 200 
LIMIT 5;

SELECT 'SHORT_DOC' as type, positive_doc, LENGTH(positive_doc) as length 
FROM '.data/processed/train_triples.parquet' 
WHERE LENGTH(positive_doc) < 100 
LIMIT 3;

SELECT 'LONG_DOC' as type, positive_doc, LENGTH(positive_doc) as length 
FROM '.data/processed/train_triples.parquet' 
WHERE LENGTH(positive_doc) > 3000 
LIMIT 3;

-- Memory usage estimation (approximate)
SELECT 
    'train' as dataset,
    COUNT(*) as total_records,
    SUM(LENGTH(query) + LENGTH(positive_doc) + LENGTH(negative_doc)) as total_text_bytes,
    ROUND(SUM(LENGTH(query) + LENGTH(positive_doc) + LENGTH(negative_doc)) / 1024.0 / 1024.0, 2) as total_text_mb
FROM '.data/processed/train_triples.parquet';

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