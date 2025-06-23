 ## MLX8-W2: Two Tower Document Search
 https://hackmd.io/@-OWUHe7_QquNsx0fCrE8Qg/rJSMl2W4le

### 📁 Perceptron Party

> #### AJ  
> #### Uncle Ben
> #### Charles
> #### Maria 

---

## Model Structure

```
two_tower_best_vivid-sweep-1_20250620_062217.pt
├── 🧠 model_state_dict                    # PyTorch model weights & biases
│   ├── query_tower.layers.0.weight       # Query encoder layer 1 weights
│   ├── query_tower.layers.0.bias         # Query encoder layer 1 biases  
│   ├── query_tower.layers.1.weight       # Query encoder layer 2 weights
│   ├── query_tower.layers.1.bias         # Query encoder layer 2 biases
│   ├── document_tower.layers.0.weight    # Document encoder layer 1 weights
│   ├── document_tower.layers.0.bias      # Document encoder layer 1 biases
│   ├── document_tower.layers.1.weight    # Document encoder layer 2 weights
│   ├── document_tower.layers.1.bias      # Document encoder layer 2 biases
│   └── embedding.weight                  # Word embeddings (vocab_size × embed_dim)
│
├── 📚 embedding_matrix                    # Pre-trained embeddings tensor [vocab_size, embed_dim]
├── 🔤 word_to_index                      # Dict: {"word": index, ...}
├── 🔢 index_to_word                      # Dict: {index: "word", ...}
│
├── ⚙️  model_config                       # Model architecture parameters
│   ├── hidden_dim: 256                   # Hidden layer dimension
│   ├── num_layers: 2                     # Number of encoder layers
│   └── dropout: 0.1                     # Dropout rate
│
├── 📊 training_stats                     # Complete training history
│   ├── 🎛️  config                        # FULL SWEEP CONFIGURATION
│   │   ├── batch_size: 32               # Training batch size
│   │   ├── learning_rate: 0.0001        # Learning rate
│   │   ├── optimizer: "adamw"           # Optimizer type
│   │   ├── weight_decay: 1e-05          # L2 regularization
│   │   ├── beta1: 0.9                   # Adam beta1
│   │   ├── beta2: 0.999                 # Adam beta2
│   │   ├── loss_type: "contrastive"     # Loss function type
│   │   ├── temperature: 0.1             # Contrastive loss temperature
│   │   ├── margin: 0.2                  # Margin for triplet/margin losses
│   │   ├── scheduler: "cosine"          # Learning rate scheduler
│   │   ├── epochs: 5                    # Total training epochs
│   │   ├── hidden_dim: 256              # Architecture: hidden dimension
│   │   ├── num_layers: 2                # Architecture: number of layers
│   │   ├── dropout: 0.1                 # Architecture: dropout rate
│   │   ├── patience: 3                  # Early stopping patience
│   │   ├── grad_clip: 1.0               # Gradient clipping threshold
│   │   ├── eval_every_epoch: true       # Evaluation frequency
│   │   ├── eval_max_queries: 1000       # Max queries for evaluation
│   │   ├── triplets_file: "path/file"   # Training data path
│   │   ├── checkpoint_dir: "./checkpoints" # Model save directory
│   │   ├── evaluation_path: "./evaluation"  # Evaluation module path
│   │   └── num_workers: 0               # Data loading workers
│   │
│   ├── 📈 training_stats[]              # Per-epoch training metrics
│   │   ├── [epoch 1]
│   │   │   ├── epoch: 1
│   │   │   ├── train_loss: 0.245
│   │   │   ├── train_accuracy: 0.673
│   │   │   ├── learning_rate: 0.0001
│   │   │   ├── eval_mrr: 0.234
│   │   │   ├── eval_mrr@10: 0.234
│   │   │   ├── eval_recall@10: 0.456
│   │   │   ├── eval_ndcg@10: 0.312
│   │   │   └── eval_map: 0.198
│   │   ├── [epoch 2] {...}
│   │   └── [epoch N] {...}
│   │
│   ├── 📊 evaluation_stats[]            # Per-epoch detailed evaluation
│   │   ├── [epoch 1]
│   │   │   ├── MRR: 0.234
│   │   │   ├── MRR@10: 0.234
│   │   │   ├── Recall@10: 0.456
│   │   │   ├── NDCG@10: 0.312
│   │   │   ├── MAP: 0.198
│   │   │   └── evaluation_type: "full_eval"
│   │   └── [epoch N] {...}
│   │
│   ├── 🏆 best_metrics                   # Best evaluation scores
│   │   ├── MRR: 0.3834                  # Best Mean Reciprocal Rank
│   │   ├── MRR@10: 0.3834               # Best MRR@10
│   │   ├── Recall@10: 0.6789            # Best Recall@10
│   │   ├── NDCG@10: 0.4567              # Best NDCG@10
│   │   └── MAP: 0.2891                  # Best Mean Average Precision
│   │
│   └── 🥇 best_epoch: 3                 # Epoch with best performance
│
├── 📅 epoch: 3                          # Final epoch number
├── 📊 eval_results                      # Final evaluation results (same as best_metrics)
└── 🏷️  wandb_run_id: "vivid-sweep-1"    # Weights & Biases run identifier
```


---

## Caching Documents into Redis Vector DB

```
Sample document IDs from SQLite:
  19699_0: Since 2007, the RBA's outstanding reputation has been affect...
  19699_1: The Reserve Bank of Australia (RBA) came into being on 14 Ja...
  19699_2: RBA Recognized with the 2014 Microsoft US Regional Partner o...
  19699_3: The inner workings of a rebuildable atomizer are surprisingl...
  19699_4: Results-Based Accountability® (also known as RBA) is a disci...








```


---

## Evaluation

```
"evaluation_stats": [
    {
      "num_queries_evaluated": 484,
      "num_queries_with_relevance": 484,
      "MRR": 0.34510937295028205,
      "MRR@10": 0.34510937295028205,
      "MAP": 0.34510937295028205,
      "Recall@1": 0.12603305785123967,
      "NDCG@1": 0.12603305785123967,
      "Recall@5": 0.6714876033057852,
      "NDCG@5": 0.38945517026146864,
      "Recall@10": 1.0,
      "NDCG@10": 0.4986663919672747,
      "Recall@20": 1.0,
      "NDCG@20": 0.4986663919672747,
      "epoch": 1
    },
```


---

## Evaluation II

+ Alt model using CBOW embeds from Week 1

+ Two Tower (GRU) NN

```
"evaluation_stats": [
    {
      "Recall@10": 0.2241, (solid according to ChatGPT 😊)
      "Recall@5": 0.0131,
      "Recall@3": 0.0020,
    },
```

---

# 🤔
![Screenshot 2025-06-20 at 15.38.48](https://hackmd.io/_uploads/rk4KMgmNgx.png)

---

## Evaluation III

```
================================================================================
🏆 CHECKPOINT ANALYSIS RESULTS
================================================================================
📊 Metric used: MRR
📁 Total checkpoints: 5
📈 Checkpoints with metrics: 5

🥇 TOP 5 CHECKPOINTS:
------------------------------------------------------------------------------------------------------------------------
#   Checkpoint                               MRR      Recall@1 Recall@5 Recall@10  Size(MB)  Modified
------------------------------------------------------------------------------------------------------------------------
1   two_tower_best_true-sweep-1_20250620_084640.pt   0.4118   0.1667   0.7292   1.0000     185.2     2025-06-20 08:46
    📋 Training history: training_history_two_tower_best_true-sweep-1_20250620_084640.json
    🔄 Sweep: true-sweep-1_20250620

2   two_tower_model_20250619_085100.pt         0.3755   0.1250   0.7292   1.0000     190.7     2025-06-19 08:51
    📋 Training history: training_history_two_tower_model_20250619_085100.json

3   two_tower_best_true-sweep-1_20250620_065748.pt   0.3630   0.1458   0.6458   1.0000     185.2     2025-06-20 06:57
    📋 Training history: training_history_two_tower_best_true-sweep-1_20250620_065748.json
    🔄 Sweep: true-sweep-1_20250620

4   two_tower_best_epoch1_20250619_101644.pt   0.3451   0.1042   0.7083   1.0000     190.7     2025-06-19 10:16
    📋 Training history: training_history_two_tower_best_epoch1_20250619_101644.json
    🔄 Sweep: epoch1_20250619

5   two_tower_model_20250619_080117.pt         0.3237   0.1042   0.6458   1.0000     190.7     2025-06-19 08:01
    📋 Training history: training_history_two_tower_model_20250619_080117.json
```

---

## Model Permance (Charles)

Recall@50: 0.74
Recall@10: 0.23 ~ 0.45 
(lost track of hyper parameters, strangely MLP >> RNN)

![image](https://hackmd.io/_uploads/BJmcW-XEle.png)

---

### Inference (Charles)

Query: "cooking egg"
> 2025-06-20 15:26:40 --INFO-- Faiss search took 0.70 ms on GPU

- 1 [ **0.9026**, #12186]: 1 Remove the egg: ...

- 2 [**0.9021**, #11718]: 2 eggs. Pick and ...

- 3 [**0.8992**, #12104]: Rinse the pork, ...
- 4 [**0.8991**, #12066]: That's right, ... kitchen
- 5 [**0.8984**, #12186]: It ... 10 minutes ... an egg,
- ...

---

A colorful / colorcoded Terminal UI :100:
![image](https://hackmd.io/_uploads/Skxa1bWmNlx.png)

---


A colorful / colorcoded Terminal UI :100: 

![image](https://hackmd.io/_uploads/rJDteZQVxl.png)

---

## 🔍 Inference: Query 1 – _"How to train your model"_

**Top Results:**
- 📄 Use of interferon-gamma release assays (0.936)  
- 📄 Blood test metrics like PT and INR (0.935)  
- 📄 Facebook password reset guide (0.935)  

**⚠️ Observation:**  
Results are **off-topic** — medical and tech support content unrelated to model training.

---

## 🔍 Inference: Query 2 – _"Cooking an egg 🍳"_

**Top Results:**
- ✅ Cooking with olive oil and flour (0.933)  
- ✅ Soup recipe with ham and bay leaves (0.931)  
- ❌ Watermelon calorie info (0.931)  
- ❌ Wendy’s sandwich nutrition (0.930)

**✅ Observation:**  
Top results are **mostly relevant**, showing better alignment for everyday queries.

---

## 📊 What Did We Learn?

- ❌ Struggles with technical queries.
- ✅ Performs better on practical, simple tasks.
- 🔁 Retrieval quality is **heavily data-dependent**.

---


Live Demos are idiotic
*(but we do it anyways)*

> uv run vector_db/unified_search.py --checkpoint $BEST_CHECKPOINT --config vector_db/redis_config.json --db data/documents.db --query "Cooking an egg"

