# MS MARCO Document Search Training Pipeline

This repository contains scripts for training document search models using the MS MARCO dataset. The pipeline consists of two main components: CBOW text embeddings training and two-tower training.

## 🚀 Overview

The training pipeline follows these steps:
1. **Text Embeddings**: Train CBOW word embeddings on MS MARCO data
2. **Triplet Generation**: Generate query-document triplets for training
3. **Two-Tower Training**: Train a two-tower RNN-based retrieval model using the triplets

## 📁 Project Structure

```
├── text_embeddings/
│   └── train_text_embeddings_cbow_msmarco.py   # CBOW embeddings training
├── preprocessing/
│   └── generate_triplets.py                    # Triplet generation
├── training/
│   └── train_two_tower_from_triplets.py       # Two-tower model training
├── checkpoints/                               # Model checkpoints
└── data/                                      # Cached datasets and triplets
```

## 🔧 Prerequisites

### Required Dependencies
```bash
# Install dependencies using uv
uv pip install torch transformers datasets gensim tqdm numpy pickle-mixin
```

### Alternative: Using uv run (Recommended)
```bash
# uv can automatically manage dependencies when running scripts
# No need to install dependencies manually - uv will handle them
```

### Optional Dependencies
- **CUDA**: For GPU acceleration
- **MPS**: For Apple Silicon acceleration

## 📚 Script Documentation

### 1. CBOW Text Embeddings Training

**File**: `text_embeddings/train_text_embeddings_cbow_msmarco.py`

Trains CBOW (Continuous Bag of Words) embeddings using the MS MARCO dataset.

#### Usage
```bash
# Basic usage (saves to ./checkpoints/)
uv run text_embeddings/train_text_embeddings_cbow_msmarco.py

# Custom checkpoint directory
uv run text_embeddings/train_text_embeddings_cbow_msmarco.py -c ./my_models/

# View help
uv run text_embeddings/train_text_embeddings_cbow_msmarco.py --help
```

#### Parameters
- `--checkpoint-dir`, `-c` (default: `./checkpoints`)
  - Directory to save model checkpoints

#### Features
- **Automatic Caching**: Downloads and caches MS MARCO dataset in `./data/`
- **Smart Preprocessing**: Caches processed texts to avoid reprocessing
- **Training Timing**: Tracks and reports training duration
- **Directory Creation**: Automatically creates checkpoint directories

#### Output
- **Model File**: `{checkpoint_dir}/msmarco_word2vec.pt`
- **Contents**: Embedding matrix, word-to-index mapping, index-to-word mapping

#### Example Output
```
Starting CBOW implementation with MS Marco dataset
Checkpoints will be saved to: ./checkpoints
✅ Dataset loaded (cached in: ./data)
📦 Loading cached processed texts from ./data/processed_texts.pkl...
✅ Loaded 45,123 cached text documents
Training completed in 245.67 seconds (4.09 minutes)

=== Model Statistics ===
Vocabulary size: 15,432
Embedding dimension: 300
Total parameters: 4,629,600
Training time: 245.67 seconds (4.09 minutes)
```

### 2. Two-Tower Model Training

**File**: `training/train_two_tower_from_triplets.py`

Trains a two-tower retrieval model using pre-generated triplets and CBOW embeddings.

#### Prerequisites
1. **CBOW Embeddings**: Must have `msmarco_word2vec.pt` file in the checkpoint directory
2. **Triplets**: Must have generated triplets using `preprocessing/generate_triplets.py`

#### Usage
```bash
# Basic usage
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_20241201_143022.pkl

# With custom parameters
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_20241201_143022.pkl \
  --epochs 3 \
  --batch-size 64 \
  --learning-rate 2e-4 \
  --num-workers 4

# With custom checkpoint directory
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_20241201_143022.pkl \
  --checkpoint-dir ./my_models/

# Training without saving model
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_20241201_143022.pkl \
  --no-save
```

#### Parameters
- **Required**:
  - `triplets_file`: Path to pre-generated triplets file (.pkl)

- **Optional**:
  - `--epochs` (default: 1): Number of training epochs
  - `--batch-size` (default: 32): Batch size for training
  - `--learning-rate` (default: 1e-4): Learning rate
  - `--num-workers` (default: 0): DataLoader workers (0 for safety)
  - `--checkpoint-dir`, `-c` (default: `./checkpoints`): Directory for model checkpoints
  - `--no-save`: Don't save the trained model

#### Features
- **Multi-Device Support**: Automatically detects CUDA/MPS/CPU
- **Progress Tracking**: Real-time training metrics
- **Model Saving**: Saves model state, embeddings, and training logs
- **Quick Testing**: Performs similarity tests after training

#### Output Files
- **Model**: `{checkpoint_dir}/two_tower_model_{timestamp}.pt`
- **Training Log**: `{checkpoint_dir}/two_tower_model_{timestamp}_training_log.json`

## 🔄 Complete Workflow

### Option 1: Default Directories

### Step 1: Train CBOW Embeddings
```bash
# Train embeddings (saves to ./checkpoints/)
uv run text_embeddings/train_text_embeddings_cbow_msmarco.py
```

### Step 2: Generate Triplets
```bash
# Generate training triplets (saves to ./data/)
uv run preprocessing/generate_triplets.py --max-samples 50000
```

### Step 3: Train Two-Tower Model
```bash
# Train two-tower model (uses embeddings from ./checkpoints/, saves model to ./checkpoints/)
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_*.pkl --epochs 3
```

### Option 2: Custom Checkpoint Directory

```bash
# Step 1: Train CBOW embeddings to custom directory
uv run text_embeddings/train_text_embeddings_cbow_msmarco.py -c ./my_models/

# Step 2: Generate triplets (same as before)
uv run preprocessing/generate_triplets.py --max-samples 50000

# Step 3: Train two-tower model using same custom directory
uv run training/train_two_tower_from_triplets.py ./data/msmarco_triplets_50k_*.pkl -c ./my_models/ --epochs 3
```

## 📊 Expected File Structure After Training

```
├── checkpoints/
│   ├── msmarco_word2vec.pt                    # CBOW embeddings
│   ├── two_tower_model_20241201_150234.pt     # Trained two-tower model
│   └── two_tower_model_20241201_150234_training_log.json
├── data/
│   ├── processed_texts.pkl                    # Cached processed texts
│   ├── msmarco_triplets_50k_20241201_143022.pkl   # Training triplets
│   └── msmarco_triplets_50k_20241201_143022_metadata.json
```

## 🎯 Performance Tips

### For CBOW Training
- Uses all available CPU cores by default
- Caching significantly speeds up subsequent runs
- Larger `max_samples` = better embeddings but longer training

### For Two-Tower Training
- Use `--num-workers 0` to avoid multiprocessing issues
- Increase `--batch-size` if you have enough GPU memory
- Monitor GPU utilization and adjust batch size accordingly

## 🔍 Troubleshooting

### Common Issues

**1. "File not found" errors**
- Ensure you've run the CBOW training before two-tower training
- Check that triplets file path is correct
- If using custom checkpoint directories, ensure both scripts use the same directory

**2. Memory issues**
- Reduce `--batch-size` for two-tower training
- Use `--num-workers 0` to avoid memory duplication

**3. Slow training**
- Enable GPU acceleration (CUDA/MPS)
- Increase `--num-workers` for faster data loading (if no memory issues)

### Verification Commands
```bash
# Check if CBOW embeddings exist
ls -la checkpoints/msmarco_word2vec.pt

# Check available triplets
ls -la data/msmarco_triplets_*.pkl

# Verify two-tower model output
ls -la checkpoints/two_tower_model_*.pt
```

## 📈 Expected Training Times

| Component | Dataset Size | Time (CPU) | Time (GPU) |
|-----------|-------------|------------|------------|
| CBOW Embeddings | 50k samples | ~4 minutes | ~2 minutes |
| Two-Tower (1 epoch) | 50k triplets | ~10 minutes | ~3 minutes |
| Two-Tower (3 epochs) | 50k triplets | ~30 minutes | ~8 minutes |

*Times are approximate and depend on hardware specifications.*

## 📝 Notes

- All scripts support `--help` for detailed parameter information
- Caching is enabled by default to speed up subsequent runs
- Models are saved with timestamps to avoid overwriting
- Training logs include detailed metrics for analysis 

## 📁 Data Path Configuration

When running scripts from the `.ben` folder, the data path configuration is designed to be consistent and flexible:

### Default Data Path
- **Default**: `./data` (current directory + data folder)
- **Location**: When running from `.ben/`, this resolves to `.ben/data/`
- **Override**: Use `--data` or `--data-path` arguments to specify a different location

### Updated Scripts
The following scripts have been updated to use `./data` as the default data path:

1. **Vector Database Scripts** (`.ben/vector_db/`):
   - `example_usage.py` - Example configurations for Redis vector store
   - `document_store.py` - SQLite document store operations

2. **Training Scripts** (`.ben/training/`):
   - `train_with_wandb.py` - W&B training with triplets
   - `two_tower_model.py` - Model definitions and training ✅ (already correct)

3. **Preprocessing Scripts** (`.ben/preprocessing/`):
   - `generate_triplets.py` - Generate MS MARCO triplets ✅ (already correct)

4. **Evaluation Scripts** (`.ben/evaluation/`):
   - `ms_marco_evaluator.py` - Model evaluation ✅ (already correct)

### Usage Examples

```bash
# Run from .ben folder - uses ./data (which is .ben/data)
cd .ben
python vector_db/document_store.py

# Override data path if needed
python vector_db/document_store.py --data ../some_other_data

# Training with default data path
python training/train_with_wandb.py --triplets-file ./data/msmarco_triplets_5k.pkl

# Generate triplets with custom data path
python preprocessing/generate_triplets.py --data-path /path/to/custom/data
```

### Data Directory Structure
```
.ben/
├── data/                                    # ← Default data location
│   ├── msmarco_validation_eval.pkl
│   ├── msmarco_triplets_*.pkl
│   ├── msmarco_raw_data*.pkl
│   └── processed_texts.pkl
├── vector_db/
├── training/
├── preprocessing/
└── evaluation/
```

### Benefits
- **Consistent**: All scripts use the same `./data` default
- **Flexible**: Easy to override with command-line arguments
- **Clear**: Help text indicates default path behavior
- **Portable**: Works regardless of where you run the scripts from 