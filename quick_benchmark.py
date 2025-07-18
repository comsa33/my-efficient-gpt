"""
Quick benchmark to test efficiency improvements across different text types.
"""

import os
import time
import torch
import numpy as np
import pickle

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


def quick_benchmark():
    """Run quick benchmarks on available datasets."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = []
    
    # Test configurations
    datasets = [
        {
            'name': 'Shakespeare (Character-level)',
            'path': 'data/shakespeare_char',
            'vocab_size': 65,
            'description': 'Classic literature with archaic English'
        }
    ]
    
    # Add Shakespeare BPE if available
    if os.path.exists('data/shakespeare/train.bin'):
        datasets.append({
            'name': 'Shakespeare (BPE)',
            'path': 'data/shakespeare', 
            'vocab_size': 50257,
            'description': 'Same text with subword tokenization'
        })
    
    # Model configurations
    configs = {
        'small': {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'block_size': 128},
        'medium': {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'block_size': 256}
    }
    
    print("\n" + "="*80)
    print("QUICK EFFICIENCY BENCHMARK")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n📚 Dataset: {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        
        # Load data
        train_data = np.memmap(os.path.join(dataset['path'], 'train.bin'), 
                              dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(dataset['path'], 'val.bin'), 
                            dtype=np.uint16, mode='r')
        
        for size_name, size_config in configs.items():
            print(f"\n   Model size: {size_name}")
            
            # Base config
            base_config = {
                'vocab_size': dataset['vocab_size'],
                'dropout': 0.1,
                'bias': False,
                **size_config
            }
            
            # Test standard model
            print("   - Standard GPT: ", end='', flush=True)
            config = GPTConfig(**base_config)
            model = GPT(config).to(device)
            
            # Quick training
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            batch_size = 16
            train_iters = 100
            
            start_time = time.time()
            model.train()
            losses = []
            
            for i in range(train_iters):
                ix = torch.randint(len(train_data) - config.block_size, (batch_size,))
                x = torch.stack([torch.from_numpy((train_data[j:j+config.block_size]).astype(np.int64)) for j in ix])
                y = torch.stack([torch.from_numpy((train_data[j+1:j+1+config.block_size]).astype(np.int64)) for j in ix])
                x, y = x.to(device), y.to(device)
                
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            train_time = time.time() - start_time
            avg_loss = np.mean(losses[-20:])
            
            # Test inference speed
            model.eval()
            tokens_processed = 0
            inference_start = time.time()
            
            with torch.no_grad():
                for _ in range(20):
                    test_x = torch.randint(0, dataset['vocab_size'], 
                                         (8, config.block_size)).to(device)
                    _ = model(test_x)
                    tokens_processed += 8 * config.block_size
            
            inference_time = time.time() - inference_start
            std_tokens_per_sec = tokens_processed / inference_time
            
            print(f"Loss: {avg_loss:.3f}, Speed: {std_tokens_per_sec:.0f} tokens/sec")
            
            # Test efficient model
            print("   - Efficient GPT: ", end='', flush=True)
            
            efficiency = EfficiencyConfig(
                enable_early_exit=True,
                exit_threshold=0.9,
                min_exit_layer=max(1, size_config['n_layer'] // 3),
                enable_sparse_activation=True,
                sparsity_ratio=0.1,
                enable_local_attention=True,
                local_attention_window=min(64, size_config['block_size'] // 2),
                track_efficiency_metrics=True
            )
            
            eff_config = EfficientGPTConfig(**base_config, efficiency=efficiency)
            eff_model = EfficientGPT(eff_config).to(device)
            
            # Quick training
            eff_optimizer = torch.optim.AdamW(eff_model.parameters(), lr=3e-4)
            
            start_time = time.time()
            eff_model.train()
            eff_losses = []
            
            for i in range(train_iters):
                ix = torch.randint(len(train_data) - config.block_size, (batch_size,))
                x = torch.stack([torch.from_numpy((train_data[j:j+config.block_size]).astype(np.int64)) for j in ix])
                y = torch.stack([torch.from_numpy((train_data[j+1:j+1+config.block_size]).astype(np.int64)) for j in ix])
                x, y = x.to(device), y.to(device)
                
                logits, loss = eff_model(x, y)
                eff_optimizer.zero_grad()
                loss.backward()
                eff_optimizer.step()
                eff_losses.append(loss.item())
            
            eff_train_time = time.time() - start_time
            eff_avg_loss = np.mean(eff_losses[-20:])
            
            # Test inference speed
            eff_model.eval()
            tokens_processed = 0
            inference_start = time.time()
            
            with torch.no_grad():
                for _ in range(20):
                    test_x = torch.randint(0, dataset['vocab_size'], 
                                         (8, config.block_size)).to(device)
                    _ = eff_model(test_x)
                    tokens_processed += 8 * config.block_size
            
            inference_time = time.time() - inference_start
            eff_tokens_per_sec = tokens_processed / inference_time
            
            # Get efficiency stats
            if hasattr(eff_model, 'get_efficiency_summary'):
                stats = eff_model.get_efficiency_summary()
            else:
                stats = {}
            
            print(f"Loss: {eff_avg_loss:.3f}, Speed: {eff_tokens_per_sec:.0f} tokens/sec")
            
            # Calculate improvements
            speedup = eff_tokens_per_sec / std_tokens_per_sec
            loss_diff = ((eff_avg_loss - avg_loss) / avg_loss) * 100
            
            print(f"   → Speedup: {speedup:.2f}x, Loss difference: {loss_diff:+.1f}%")
            
            if stats:
                print(f"   → Efficiency stats: ", end='')
                stat_strs = []
                if stats.get('avg_exit_layer'):
                    stat_strs.append(f"Exit at layer {stats['avg_exit_layer']:.1f}/{size_config['n_layer']}")
                if stats.get('avg_sparsity'):
                    stat_strs.append(f"Sparsity {stats['avg_sparsity']:.1%}")
                print(", ".join(stat_strs))
            
            # Store results
            results.append({
                'dataset': dataset['name'],
                'model_size': size_name,
                'speedup': speedup,
                'loss_diff': loss_diff,
                'std_speed': std_tokens_per_sec,
                'eff_speed': eff_tokens_per_sec,
                'efficiency_stats': stats
            })
            
            # Clear GPU cache
            if device == 'cuda':
                del model, eff_model
                torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Average speedup by dataset
    print("\nAverage Speedup by Dataset:")
    dataset_speedups = {}
    for r in results:
        if r['dataset'] not in dataset_speedups:
            dataset_speedups[r['dataset']] = []
        dataset_speedups[r['dataset']].append(r['speedup'])
    
    for dataset, speedups in dataset_speedups.items():
        avg_speedup = np.mean(speedups)
        print(f"- {dataset}: {avg_speedup:.2f}x")
    
    # Average speedup by model size
    print("\nAverage Speedup by Model Size:")
    size_speedups = {}
    for r in results:
        if r['model_size'] not in size_speedups:
            size_speedups[r['model_size']] = []
        size_speedups[r['model_size']].append(r['speedup'])
    
    for size, speedups in size_speedups.items():
        avg_speedup = np.mean(speedups)
        print(f"- {size}: {avg_speedup:.2f}x")
    
    # Overall average
    all_speedups = [r['speedup'] for r in results]
    overall_avg = np.mean(all_speedups)
    print(f"\n🎯 Overall Average Speedup: {overall_avg:.2f}x")
    
    # Generate markdown table for README
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR README")
    print("="*80)
    print("\n```markdown")
    print("| Dataset | Model Size | Standard (tokens/sec) | Efficient (tokens/sec) | Speedup | Quality Impact |")
    print("|---------|------------|-----------------------|------------------------|---------|----------------|")
    
    for r in results:
        quality = "< 2%" if abs(r['loss_diff']) < 2 else f"{r['loss_diff']:+.1f}%"
        print(f"| {r['dataset']} | {r['model_size'].capitalize()} | "
              f"{r['std_speed']:.0f} | {r['eff_speed']:.0f} | "
              f"{r['speedup']:.2f}x | {quality} |")
    
    print(f"\n**Average Speedup: {overall_avg:.2f}x** 🚀")
    print("```")
    
    return results


if __name__ == "__main__":
    quick_benchmark()