# 🔍 Checkpoint Finder Scripts

This directory contains scripts to automatically find and identify the best performing checkpoint from your training runs.

## 📁 Files

- **`find_best_checkpoint.py`** - Main analysis script
- **`set_best_checkpoint.sh`** - Shell script to set environment variables
- **`README_checkpoint_finder.md`** - This documentation

## 🚀 Quick Start

### Find Best Checkpoint (Interactive)
```bash
cd .ben
python find_best_checkpoint.py
```

### Get Best Checkpoint Path Only
```bash
cd .ben
python find_best_checkpoint.py --best-only
```

### Set Environment Variables
```bash
cd .ben
source set_best_checkpoint.sh
echo $BEST_CHECKPOINT
```

## 📊 Script Features

### `find_best_checkpoint.py`

**What it does:**
- Scans all `.pt` checkpoint files in `./checkpoints/`
- Parses training history JSON files for performance metrics
- Analyzes W&B configuration files
- Applies known best checkpoint information from previous analysis
- Ranks checkpoints by performance metrics (MRR, MAP, NDCG, accuracy)

**Usage:**
```bash
python find_best_checkpoint.py [OPTIONS]

Options:
  --metric MRR          Metric to rank by (MRR, MAP, NDCG, accuracy) [default: MRR]
  --verbose, -v         Show detailed analysis information
  --top N, -n N         Number of top checkpoints to display [default: 5]
  --best-only           Only output the best checkpoint path (for scripting)
  --help               Show help message
```

**Examples:**
```bash
# Standard analysis
python find_best_checkpoint.py

# Verbose analysis with top 10 results
python find_best_checkpoint.py --verbose --top 10

# Rank by MAP instead of MRR
python find_best_checkpoint.py --metric MAP

# Get just the path for scripting
BEST_PATH=$(python find_best_checkpoint.py --best-only)
```

### `set_best_checkpoint.sh`

**What it does:**
- Runs the Python script to find the best checkpoint
- Sets environment variables for easy use in other scripts
- Provides both absolute and relative paths

**Environment Variables Set:**
- `BEST_CHECKPOINT` - Absolute path to best checkpoint
- `BEST_CHECKPOINT_REL` - Relative path from `.ben` directory

**Usage:**
```bash
# Source the script to set environment variables
source set_best_checkpoint.sh

# Use in other commands
python cache_from_sqlite.py --checkpoint "$BEST_CHECKPOINT_REL" --config ./redis_config.json
```

## 🏆 Current Best Checkpoint

Based on previous W&B analysis, the current best checkpoint is:

- **File**: `two_tower_best_vivid-sweep-1_20250620_062217.pt`
- **MRR**: 0.3834
- **Size**: ~191MB
- **Training Date**: June 20, 2025

## 🔧 How It Works

### 1. Checkpoint Discovery
- Scans `./checkpoints/` directory for `.pt` files
- Extracts metadata from filenames (sweep names, timestamps)
- Gets file size and modification dates

### 2. Metrics Matching
- Parses training history JSON files
- Matches checkpoints to their performance metrics by timestamp proximity
- Applies known best checkpoint information from previous analysis

### 3. Ranking Algorithm
- Sorts by specified metric (MRR by default)
- Falls back to modification time for ties
- Handles alternative metric names (e.g., `eval_mrr`, `MRR@10`)

### 4. Output Formatting
- Shows top N checkpoints with detailed information
- Highlights known best checkpoint with ⭐
- Provides export commands for easy scripting

## 📈 Supported Metrics

- **MRR** (Mean Reciprocal Rank) - Primary ranking metric
- **MAP** (Mean Average Precision)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **Accuracy** - Training/evaluation accuracy

Alternative metric names are automatically handled:
- `MRR` → `mrr`, `eval_mrr`, `MRR@10`
- `MAP` → `map`, `eval_map`
- `NDCG` → `ndcg`, `NDCG@10`, `eval_ndcg@10`
- `accuracy` → `train_accuracy`, `eval_accuracy`

## 🔄 Integration Examples

### Cache All Documents with Best Checkpoint
```bash
source set_best_checkpoint.sh
python cache_from_sqlite.py --checkpoint "$BEST_CHECKPOINT_REL" --config ./redis_config.json
```

### Run Inference with Best Checkpoint
```bash
source set_best_checkpoint.sh
python unified_search.py --checkpoint "$BEST_CHECKPOINT_REL" --config ./redis_config.json --query "your search query"
```

### Use in Training Scripts
```bash
BEST_PATH=$(python find_best_checkpoint.py --best-only)
python train_with_wandb.py --resume-from "$BEST_PATH" --triplets-file ../data/triplets.tsv
```

## 🐛 Troubleshooting

### No Metrics Found
If the script shows "📈 Checkpoints with metrics: 0":
- Check that training history JSON files exist in `./checkpoints/`
- Verify JSON files contain `best_metrics` or `evaluation_stats`
- The script will still work using known best checkpoint info

### Wrong Best Checkpoint
If you know a different checkpoint is better:
1. Edit the `known_best` dictionary in `find_best_checkpoint.py`
2. Update the filename, MRR value, and source

### Permission Issues
```bash
chmod +x set_best_checkpoint.sh
```

## 📝 Notes

- Script is designed to run from the `.ben` folder
- Automatically handles relative paths
- Skips word2vec embedding files
- Uses timestamp proximity (6 hours) to match checkpoints to training histories
- Falls back to newest checkpoint if no metrics are available 