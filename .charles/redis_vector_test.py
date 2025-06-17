import redis
import numpy as np

# Connect to Redis Stack
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Parameters
VECTOR_DIM = 4
INDEX_NAME = "vec_idx"
DOC_PREFIX = "doc:"

# Create index with vector field
try:
    r.ft(INDEX_NAME).create_index([
        redis.commands.search.field.TagField("tag"),
        redis.commands.search.field.VectorField("vec", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE"
        })
    ])
except Exception as e:
    if "Index already exists" not in str(e):
        raise

# Add sample documents
pipe = r.pipeline()
for i in range(3):
    vec = np.random.rand(VECTOR_DIM).astype(np.float32).tobytes()
    pipe.hset(f"{DOC_PREFIX}{i}", mapping={
        "tag": f"sample{i}",
        "vec": vec
    })
pipe.execute()

# Query: Find nearest vector
query_vec = np.random.rand(VECTOR_DIM).astype(np.float32).tobytes()
q = f'*=>[KNN 2 @vec $vec AS score]'
params = {"vec": query_vec}
results = r.ft(INDEX_NAME).search(q, query_params=params)
print("KNN Results:")
for doc in results.docs:
    print(doc.id, doc.tag, doc.score)
