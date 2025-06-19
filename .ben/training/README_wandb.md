# Two-Tower Training with Weights & Biases

This directory contains W&B-integrated training scripts for hyperparameter tuning and experiment tracking.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install wandb
# Or if using uv:
uv add wandb
```

### 2. Login to W&B

```bash
wandb login
```

### 3. Single Training Run

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_5000.pkl \
    --project "my-two-tower-project" \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-4
```

### 4. Hyperparameter Sweep

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_5000.pkl \
    --project "my-two-tower-project" \
    --sweep \
    --sweep-count 50
```

## 📊 Features

### Comprehensive Logging
- **Training Metrics**: Loss, accuracy, learning rate per batch/epoch
- **Evaluation Metrics**: MRR, MRR@10, Recall@10, NDCG@10, MAP
- **System Info**: Device, CUDA availability, model parameters
- **Model Artifacts**: Best models saved as W&B artifacts

### Hyperparameter Tuning
- **Bayesian Optimization**: Efficient hyperparameter search
- **Early Termination**: Hyperband algorithm stops poor runs early
- **Multiple Optimizers**: AdamW, Adam, SGD
- **Flexible Schedulers**: Cosine, Step, ReduceLROnPlateau

### Advanced Features
- **Model Watching**: Gradient and parameter tracking
- **Artifact Management**: Automatic model versioning
- **Early Stopping**: Configurable patience
- **Resume Support**: Automatic run resuming

## 🎯 Hyperparameter Sweep Configuration

### Tuned Parameters

| Parameter | Type | Range/Values | Description |
|-----------|------|--------------|-------------|
| `hidden_dim` | Discrete | [128, 256, 512, 768] | Hidden layer dimension |
| `num_layers` | Discrete | [1, 2, 3, 4] | Number of transformer layers |
| `dropout` | Continuous | [0.0, 0.4] | Dropout rate |
| `batch_size` | Discrete | [16, 32, 64, 128] | Training batch size |
| `learning_rate` | Log-uniform | [1e-5, 1e-2] | Initial learning rate |
| `weight_decay` | Log-uniform | [1e-6, 1e-2] | L2 regularization |
| `temperature` | Continuous | [0.05, 0.3] | Contrastive loss temperature |
| `optimizer` | Discrete | ["adamw", "adam"] | Optimization algorithm |
| `scheduler` | Discrete | ["cosine", "step", "reduce_on_plateau"] | LR scheduler |
| `loss_type` | Discrete | ["contrastive", "margin_contrastive", "triplet"] | Loss function type |
| `margin` | Continuous | [0.1, 0.5] | Margin for margin-based and triplet losses |

### Fixed Parameters
- `epochs`: 5 (for sweep efficiency)
- `eval_max_queries`: 1000 (evaluation speed)
- `patience`: 3 (early stopping)
- `grad_clip`: 1.0 (gradient clipping)

## 📈 Usage Examples

### 1. Basic Single Run

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_5000.pkl \
    --epochs 5 \
    --batch-size 32
```

### 2. Custom Configuration

Create `config.json`:
```json
{
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.2,
    "learning_rate": 5e-4,
    "temperature": 0.15,
    "optimizer": "adamw",
    "scheduler": "cosine"
}
```

Run with config:
```bash
python train_with_wandb.py \
    --config config.json \
    --triplets-file ./data/msmarco_triplets_5000.pkl
```

### 3. Large-Scale Hyperparameter Sweep

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_50000.pkl \
    --project "two-tower-large-sweep" \
    --sweep \
    --sweep-count 100
```

### 4. GPU Training with More Epochs

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_50000.pkl \
    --epochs 20 \
    --batch-size 128 \
    --learning-rate 1e-3
```

## 🔧 Advanced Configuration

### Custom Sweep Configuration

You can modify the sweep configuration in the `create_sweep_config()` function:

```python
def create_sweep_config() -> Dict:
    return {
        "method": "bayes",  # or "grid", "random"
        "metric": {
            "name": "eval/MRR",
            "goal": "maximize"
        },
        "parameters": {
            # Your custom parameter ranges
        }
    }
```

### Environment Variables

Set these for better integration:

```bash
export WANDB_PROJECT="two-tower-msmarco"
export WANDB_ENTITY="your-username"
export WANDB_API_KEY="your-api-key"
```

## 📊 Monitoring and Analysis

### W&B Dashboard Features

1. **Real-time Training**: Live loss and accuracy curves
2. **Hyperparameter Importance**: Which params matter most
3. **Parallel Coordinates**: Visualize parameter relationships
4. **Model Comparison**: Compare runs side-by-side
5. **Artifact Versioning**: Track model evolution

### Key Metrics to Watch

- **Primary**: `eval/MRR` (Mean Reciprocal Rank)
- **Secondary**: `eval/MRR@10`, `eval/Recall@10`, `eval/NDCG@10`
- **Training**: `train/loss`, `train/accuracy`
- **System**: `train/learning_rate`, memory usage

## 🚨 Best Practices

### 1. Start Small
```bash
# Test with small dataset first
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_1000.pkl \
    --epochs 2
```

### 2. Use Tags
```python
wandb.init(
    project="two-tower-msmarco",
    tags=["experiment-1", "baseline", "small-data"]
)
```

### 3. Monitor Resource Usage
- Check GPU utilization in W&B
- Monitor memory usage
- Watch for OOM errors in logs

### 4. Save Important Runs
- Star successful runs in W&B dashboard
- Download best model artifacts
- Export sweep results for analysis

## 🔍 Troubleshooting

### Common Issues

1. **W&B Login Failed**
   ```bash
   wandb login --relogin
   ```

2. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Reduce `hidden_dim`
   - Use gradient accumulation

3. **Slow Evaluation**
   - Reduce `eval_max_queries`
   - Set `eval_every_epoch: False`

4. **Sweep Not Starting**
   - Check W&B project permissions
   - Verify sweep configuration syntax
   - Check internet connection

### Debug Mode

```bash
python train_with_wandb.py \
    --triplets-file ./data/msmarco_triplets_100.pkl \
    --epochs 1 \
    --batch-size 8
```

## 📚 Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [Hyperparameter Sweeps Guide](https://docs.wandb.ai/guides/sweeps)
- [PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [Artifact Management](https://docs.wandb.ai/guides/artifacts)

## 🎉 Expected Results

After running sweeps, you should see:

- **MRR improvements**: 0.15-0.25 typical range
- **Optimal batch sizes**: Usually 32-64 for this model
- **Best learning rates**: Often in 1e-4 to 5e-4 range
- **Architecture**: 2-3 layers with 256-512 hidden dim often optimal

Happy hyperparameter tuning! 🚀 