#!/usr/bin/env python3
"""
Find Best Checkpoint Script

This script analyzes all available checkpoints and their performance metrics
to identify the best performing model. Designed to run from the .ben folder.

Usage:
    python find_best_checkpoint.py [--metric MRR] [--verbose]
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml


class CheckpointAnalyzer:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.wandb_dir = self.base_dir.parent / "wandb"
        
        # Known best checkpoint from previous analysis
        self.known_best = {
            'filename': 'two_tower_best_vivid-sweep-1_20250620_062217.pt',
            'mrr': 0.3834,
            'source': 'Previous W&B analysis'
        }
        
    def find_training_histories(self) -> List[Path]:
        """Find all training history JSON files"""
        patterns = [
            self.checkpoints_dir / "training_history_*.json",
            self.checkpoints_dir / "*_training_log.json"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(str(pattern)))
        
        return [Path(f) for f in files]
    
    def parse_training_history(self, file_path: Path) -> Dict:
        """Parse a training history JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            result = {
                'file': file_path.name,
                'path': str(file_path),
                'metrics': {},
                'best_epoch': data.get('best_epoch', 0),
                'config': data.get('config', {})
            }
            
            # Extract best metrics
            if 'best_metrics' in data and data['best_metrics']:
                result['metrics'] = data['best_metrics']
            elif 'evaluation_stats' in data and data['evaluation_stats']:
                # Find best evaluation stats
                best_mrr = 0
                best_stats = {}
                for stats in data['evaluation_stats']:
                    if stats.get('MRR', 0) > best_mrr:
                        best_mrr = stats['MRR']
                        best_stats = stats
                result['metrics'] = best_stats
            elif 'final_metrics' in data:
                result['metrics'] = data['final_metrics']
            
            return result
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def find_wandb_configs(self) -> List[Dict]:
        """Find W&B sweep configurations that might contain performance info"""
        if not self.wandb_dir.exists():
            return []
        
        configs = []
        for config_file in self.wandb_dir.glob("**/*.yaml"):
            if "config-" in config_file.name:
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract run info
                    run_info = {
                        'file': config_file.name,
                        'path': str(config_file),
                        'config': config
                    }
                    configs.append(run_info)
                except Exception as e:
                    continue
        
        return configs
    
    def find_checkpoint_files(self) -> List[Dict]:
        """Find all checkpoint .pt files and extract metadata"""
        if not self.checkpoints_dir.exists():
            return []
        
        checkpoints = []
        for pt_file in self.checkpoints_dir.glob("*.pt"):
            if "word2vec" in pt_file.name.lower():
                continue  # Skip word2vec embeddings
            
            # Extract metadata from filename
            filename = pt_file.name
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(pt_file.stat().st_mtime)
            
            checkpoint_info = {
                'filename': filename,
                'path': str(pt_file),
                'size_mb': round(size_mb, 1),
                'modified': modified_time,
                'sweep_name': None,
                'timestamp': None
            }
            
            # Extract sweep name and timestamp from filename
            if "_best_" in filename:
                parts = filename.split("_best_")
                if len(parts) > 1:
                    sweep_part = parts[1].replace(".pt", "")
                    if "_" in sweep_part:
                        sweep_name, timestamp = sweep_part.rsplit("_", 1)
                        checkpoint_info['sweep_name'] = sweep_name
                        checkpoint_info['timestamp'] = timestamp
            
            checkpoints.append(checkpoint_info)
        
        return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)
    
    def apply_known_best_info(self, checkpoints: List[Dict]) -> List[Dict]:
        """Apply known best checkpoint information"""
        for checkpoint in checkpoints:
            if checkpoint['filename'] == self.known_best['filename']:
                checkpoint['metrics'] = {
                    'MRR': self.known_best['mrr'],
                    'source': self.known_best['source']
                }
                checkpoint['is_known_best'] = True
            else:
                checkpoint['is_known_best'] = False
        
        return checkpoints
    
    def match_checkpoints_to_metrics(self, checkpoints: List[Dict], 
                                   training_histories: List[Dict]) -> List[Dict]:
        """Match checkpoint files to their performance metrics"""
        matched = []
        
        for checkpoint in checkpoints:
            checkpoint_match = checkpoint.copy()
            checkpoint_match['metrics'] = {}
            checkpoint_match['training_history'] = None
            
            # Try to match by timestamp or sweep name
            for history in training_histories:
                if not history:
                    continue
                
                # Simple matching by date proximity
                history_file = Path(history['path'])
                history_time = datetime.fromtimestamp(history_file.stat().st_mtime)
                
                time_diff = abs((checkpoint['modified'] - history_time).total_seconds())
                
                # If files are within 6 hours of each other, likely related
                if time_diff < 21600:  # 6 hours in seconds
                    checkpoint_match['metrics'] = history['metrics']
                    checkpoint_match['training_history'] = history['file']
                    break
            
            matched.append(checkpoint_match)
        
        return matched
    
    def rank_checkpoints(self, checkpoints: List[Dict], 
                        metric: str = "MRR") -> List[Dict]:
        """Rank checkpoints by specified metric"""
        def get_metric_value(checkpoint):
            metrics = checkpoint.get('metrics', {})
            value = metrics.get(metric, 0)
            if value == 0:
                # Try alternative metric names
                alt_names = {
                    'MRR': ['mrr', 'eval_mrr', 'MRR@10'],
                    'MAP': ['map', 'eval_map'],
                    'NDCG': ['ndcg', 'NDCG@10', 'eval_ndcg@10'],
                    'accuracy': ['train_accuracy', 'eval_accuracy']
                }
                
                if metric in alt_names:
                    for alt_name in alt_names[metric]:
                        if alt_name in metrics:
                            value = metrics[alt_name]
                            break
            
            return float(value) if value else 0.0
        
        # Sort by metric value (descending), then by modification time (newest first)
        ranked = sorted(checkpoints, 
                       key=lambda x: (get_metric_value(x), x['modified']), 
                       reverse=True)
        
        # Add rank information
        for i, checkpoint in enumerate(ranked):
            checkpoint['rank'] = i + 1
            checkpoint['metric_value'] = get_metric_value(checkpoint)
        
        return ranked
    
    def analyze_all(self, metric: str = "MRR", verbose: bool = False, quiet: bool = False) -> Dict:
        """Perform complete analysis"""
        if not quiet:
            print("🔍 Analyzing all checkpoints...")
        
        # Find all data sources
        training_histories = []
        for file_path in self.find_training_histories():
            history = self.parse_training_history(file_path)
            if history:
                training_histories.append(history)
        
        wandb_configs = self.find_wandb_configs()
        checkpoint_files = self.find_checkpoint_files()
        
        if verbose:
            print(f"Found {len(training_histories)} training histories")
            print(f"Found {len(wandb_configs)} W&B configs")
            print(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Match checkpoints to metrics first
        matched_checkpoints = self.match_checkpoints_to_metrics(
            checkpoint_files, training_histories
        )
        
        # Apply known best information after matching
        matched_checkpoints = self.apply_known_best_info(matched_checkpoints)
        
        # Rank by performance
        ranked_checkpoints = self.rank_checkpoints(matched_checkpoints, metric)
        
        return {
            'metric_used': metric,
            'total_checkpoints': len(checkpoint_files),
            'checkpoints_with_metrics': len([c for c in ranked_checkpoints if c['metric_value'] > 0]),
            'ranked_checkpoints': ranked_checkpoints,
            'training_histories': training_histories,
            'wandb_configs': wandb_configs
        }
    
    def print_results(self, analysis: Dict, top_n: int = 5):
        """Print analysis results in a nice format"""
        print("\n" + "="*80)
        print("🏆 CHECKPOINT ANALYSIS RESULTS")
        print("="*80)
        
        print(f"📊 Metric used: {analysis['metric_used']}")
        print(f"📁 Total checkpoints: {analysis['total_checkpoints']}")
        print(f"📈 Checkpoints with metrics: {analysis['checkpoints_with_metrics']}")
        
        print(f"\n🥇 TOP {top_n} CHECKPOINTS:")
        print("-" * 80)
        
        for i, checkpoint in enumerate(analysis['ranked_checkpoints'][:top_n]):
            rank = checkpoint['rank']
            filename = checkpoint['filename']
            metric_value = checkpoint['metric_value']
            size_mb = checkpoint['size_mb']
            modified = checkpoint['modified'].strftime("%Y-%m-%d %H:%M")
            
            # Add special indicator for known best
            best_indicator = " ⭐ KNOWN BEST" if checkpoint.get('is_known_best') else ""
            
            print(f"{rank:2d}. {filename}{best_indicator}")
            print(f"    📈 {analysis['metric_used']}: {metric_value:.4f}")
            print(f"    💾 Size: {size_mb}MB | 📅 Modified: {modified}")
            
            if checkpoint.get('training_history'):
                print(f"    📋 Training history: {checkpoint['training_history']}")
            
            if checkpoint.get('sweep_name'):
                print(f"    🔄 Sweep: {checkpoint['sweep_name']}")
            
            # Show source of metrics
            if checkpoint.get('metrics', {}).get('source'):
                print(f"    🔍 Source: {checkpoint['metrics']['source']}")
            
            print()
        
        # Show best checkpoint details
        if analysis['ranked_checkpoints']:
            best = analysis['ranked_checkpoints'][0]
            print("🌟 BEST CHECKPOINT DETAILS:")
            print("-" * 40)
            print(f"File: {best['filename']}")
            print(f"Path: {best['path']}")
            print(f"{analysis['metric_used']}: {best['metric_value']:.4f}")
            
            if best.get('metrics'):
                print("\nAll available metrics:")
                for key, value in best['metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Find the best checkpoint")
    parser.add_argument("--metric", default="MRR", 
                       help="Metric to rank by (MRR, MAP, NDCG, accuracy)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--top", "-n", type=int, default=5,
                       help="Number of top checkpoints to show")
    parser.add_argument("--best-only", action="store_true",
                       help="Only show the best checkpoint path")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CheckpointAnalyzer()
    
    # Perform analysis
    analysis = analyzer.analyze_all(metric=args.metric, verbose=args.verbose, quiet=args.best_only)
    
    # Print results (unless --best-only is specified)
    if not args.best_only:
        analyzer.print_results(analysis, top_n=args.top)
    
    # Return best checkpoint path for scripting
    if analysis['ranked_checkpoints']:
        best_checkpoint = analysis['ranked_checkpoints'][0]
        
        if args.best_only:
            print(best_checkpoint['path'])
        else:
            print(f"\n💡 To use the best checkpoint:")
            print(f"export BEST_CHECKPOINT='{best_checkpoint['path']}'")
        
        return best_checkpoint['path']
    
    return None


if __name__ == "__main__":
    main() 