# MS MARCO Evaluation System

This directory contains a comprehensive evaluation system for the two-tower document retrieval model.

## 📁 Files

- **`ms_marco_evaluator.py`** - Main evaluation class with MS MARCO metrics
- **`eval_integration.py`** - Integration functions for easy evaluation during/after training

## 🎯 Evaluation Metrics

The system computes standard Information Retrieval metrics:

- **MRR (Mean Reciprocal Rank)** - Primary MS MARCO metric
- **MRR@10** - MRR limited to top 10 results
- **Recall@k** - Fraction of relevant docs in top k (k=1,5,10,20)
- **NDCG@k** - Normalized Discounted Cumulative Gain (k=1,5,10,20)
- **MAP** - Mean Average Precision

## 🚀 Quick Usage

### 1. Evaluate a Trained Model

```bash
# Basic evaluation
uv run evaluation/ms_marco_evaluator.py checkpoints/two_tower_model_20241218_123456.pt

# Quick evaluation (100 queries)
uv run evaluation/eval_integration.py checkpoints/two_tower_model_20241218_123456.pt --quick

# Full evaluation with custom settings
uv run evaluation/ms_marco_evaluator.py checkpoints/two_tower_model_20241218_123456.pt \
    --split validation \
    --max-queries 5000 \
    --batch-size 128 \
    --output evaluation/results_my_model.json
```

### 2. Compare Multiple Models

```python
from evaluation.eval_integration import compare_checkpoints

checkpoints = [
    "checkpoints/model_epoch1.pt",
    "checkpoints/model_epoch2.pt", 
    "checkpoints/model_epoch3.pt"
]

results = compare_checkpoints(checkpoints, max_queries=1000)
```

### 3. Integrate into Training

```python
from evaluation.eval_integration import evaluate_trained_model

# After each epoch in your training loop:
eval_results = evaluate_trained_model(
    model=model,
    device=device,
    epoch=current_epoch,
    max_queries=1000
)

print(f"Epoch {current_epoch} MRR: {eval_results['MRR']:.4f}")
```

## 📊 Expected Performance

For reference, here are typical MS MARCO performance ranges:

| Metric | Random | Decent | Good | Excellent |
|--------|--------|--------|------|-----------|
| MRR | 0.001 | 0.15-0.25 | 0.25-0.35 | 0.35+ |
| MRR@10 | 0.001 | 0.15-0.25 | 0.25-0.35 | 0.35+ |
| Recall@10 | 0.01 | 0.30-0.50 | 0.50-0.70 | 0.70+ |

## ⚡ Performance Tips

### Speed up Evaluation:
- Use `--max-queries 1000` for faster feedback during training
- Increase `--batch-size` if you have GPU memory
- Use `--quick` flag for rapid testing

### Memory Optimization:
- Lower batch size if you get OOM errors
- Process fewer queries at once
- Use CPU if GPU memory is limited

## 🔧 Customization

### Add Custom Metrics:
```python
def _compute_query_metrics(self, rankings, relevant_ids, k_values):
    # Add your custom metric here
    metrics = super()._compute_query_metrics(rankings, relevant_ids, k_values)
    metrics['custom_metric'] = your_calculation()
    return metrics
```

### Different Datasets:
Modify `MSMarcoEvalDataset` to load different evaluation datasets while keeping the same interface.

## 📈 Integration with Training

The evaluation system is designed to integrate seamlessly with training:

1. **After Each Epoch**: Evaluate on validation set to track progress
2. **Early Stopping**: Stop training when validation metrics plateau
3. **Model Selection**: Save the best model based on MRR
4. **Hyperparameter Tuning**: Compare different configurations

## 🐛 Troubleshooting

### Common Issues:

**Import Errors:**
```bash
# Make sure you're in the right directory
cd /path/to/your/project
uv run evaluation/ms_marco_evaluator.py
```

**Memory Errors:**
```bash
# Reduce batch size and queries
uv run evaluation/ms_marco_evaluator.py model.pt --max-queries 500 --batch-size 32
```

**Slow Evaluation:**
```bash
# Use quick evaluation during development
uv run evaluation/eval_integration.py model.pt --quick
```

## 📝 Output Format

Evaluation results are saved as JSON:

```json
{
  "metrics": {
    "MRR": 0.2845,
    "MRR@10": 0.2845, 
    "Recall@1": 0.1823,
    "Recall@10": 0.5234,
    "NDCG@10": 0.3456,
    "MAP": 0.2567
  },
  "evaluation_timestamp": "2024-12-18T12:34:56",
  "model_info": {
    "hidden_dim": 256,
    "num_layers": 2,
    "vocab_size": 50000
  }
}
```

This comprehensive evaluation system will help you track model performance and make data-driven decisions during training! 🎯 