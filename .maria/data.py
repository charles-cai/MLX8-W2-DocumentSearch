# --- Imports and NLTK setup ---
from datasets import load_dataset
import random
import pickle
import re
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

# --- Text cleaning with lemmatization ---
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    # Normalize, lowercase, remove URLs/emails/tags/punct, lemmatize
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# --- Load MS MARCO dataset (limit to MAX_EXAMPLES) ---
ds = load_dataset("microsoft/ms_marco", "v1.1")
MAX_EXAMPLES = 500_000
train_ds = ds["train"]
MAX_EXAMPLES = min(500_000, len(train_ds))
train_ds = train_ds.select(range(MAX_EXAMPLES))
print(f"✅ Using {len(train_ds)} examples from MS MARCO")

# --- Extract and clean queries and docs from MS MARCO ---
def extract_queries_and_docs(dataset):
    queries, docs = [], []
    for item in dataset:
        query = clean_text(item["query"])
        doc = ""
        if "passages" in item and isinstance(item["passages"], dict):
            passage_texts = item["passages"].get("passage_text", [])
            if passage_texts:
                doc = clean_text(passage_texts[0])
        queries.append(query)
        docs.append(doc)
    return queries, docs

# --- Extract hard negatives if available ---
def extract_hard_negatives(dataset):
    hard_negs = []
    for item in dataset:
        hn_text = ""
        if "hard_negative_passages" in item and isinstance(item["hard_negative_passages"], dict):
            texts = item["hard_negative_passages"].get("passage_text", [])
            if texts:
                hn_text = clean_text(texts[0])
        hard_negs.append(hn_text)
    return hard_negs

queries, docs = extract_queries_and_docs(train_ds)
hard_negatives = extract_hard_negatives(train_ds)

# --- Remove duplicate docs for negative sampling ---
doc_set = list(set(docs))

# --- Build (query, pos_doc, neg_doc) triples ---
def generate_triples(queries, docs, hard_negs):
    triples = []
    n = len(queries)
    for i in range(n):
        query = queries[i]
        pos_doc = docs[i]
        neg_doc = hard_negs[i]
        if not neg_doc or neg_doc == pos_doc:
            # Fallback: random negative doc
            candidates = [doc for doc in doc_set if doc != pos_doc]
            neg_doc = random.choice(candidates)
        triples.append((query, pos_doc, neg_doc))
    return triples

triples = generate_triples(queries, docs, hard_negatives)

# --- Save raw triples (cleaned strings) ---
with open("raw_train_triples.pkl", "wb") as f:
    pickle.dump(triples, f)
print(f"✅ Saved {len(triples)} triples to raw_train_triples.pkl")

# --- Save unique candidate docs for retrieval (as strings) ---
candidate_docs = list(set([pos for _, pos, _ in triples] + [neg for _, _, neg in triples]))
with open("raw_candidate_doc_texts.pkl", "wb") as f:
    pickle.dump(candidate_docs, f)
print(f"✅ Saved {len(candidate_docs)} docs to raw_candidate_doc_texts.pkl")

# --- Tokenization utilities (convert text to indices using text8 vocab) ---
def tokenize_with_vocab(text, word_to_index):
    return [word_to_index.get(word, 0) for word in text.split()]

# --- Load text8 vocab for tokenization ---
with open("text8_word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# --- Tokenize triples and save ---
tokenized_triples = []
for query, pos_doc, neg_doc in triples:
    tokenized_query = tokenize_with_vocab(query, word_to_index)
    tokenized_pos = tokenize_with_vocab(pos_doc, word_to_index)
    tokenized_neg = tokenize_with_vocab(neg_doc, word_to_index)
    tokenized_triples.append((tokenized_query, tokenized_pos, tokenized_neg))
with open("tokenized_train_triples.pkl", "wb") as f:
    pickle.dump(tokenized_triples, f)
print(f"✅ Saved {len(tokenized_triples)} tokenized triples to tokenized_train_triples.pkl")

# --- Tokenize candidate docs and save ---
tokenized_candidate_docs = [tokenize_with_vocab(doc, word_to_index) for doc in candidate_docs]
with open("tokenized_candidate_docs.pkl", "wb") as f:
    pickle.dump(tokenized_candidate_docs, f)
print(f"✅ Saved {len(tokenized_candidate_docs)} tokenized candidate docs to tokenized_candidate_docs.pkl")

# --- Preview a sample triple and candidate doc ---
print("First triple:", triples[0])
print("First candidate doc:", candidate_docs[0])
# --- Preview ---
print("First triple:", triples[0])
print("First candidate doc:", candidate_docs[0])
