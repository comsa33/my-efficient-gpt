"""
Comprehensive benchmark across different datasets and model configurations.
Tests the efficiency improvements on various types of text data.
"""

import os
import time
import json
import torch
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import subprocess

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


@dataclass
class BenchmarkResult:
    dataset: str
    model_type: str
    vocab_type: str
    train_loss: float
    val_loss: float
    train_time: float
    tokens_per_sec: float
    memory_mb: float
    efficiency_stats: Dict
    sample_text: str


class DatasetBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
        
    def prepare_shakespeare_bpe(self):
        """Prepare Shakespeare dataset with BPE tokenization."""
        print("\n📚 Preparing Shakespeare (BPE)...")
        if not os.path.exists('data/shakespeare/train.bin'):
            os.chdir('data/shakespeare')
            subprocess.run(['python', 'prepare.py'], check=True)
            os.chdir('../..')
        
        # BPE uses GPT-2 vocab size
        return 50257, 'bpe'
    
    def prepare_openwebtext_sample(self):
        """Prepare a sample of OpenWebText (would be too large for quick test)."""
        print("\n📚 Preparing OpenWebText sample...")
        # For demo purposes, we'll create a small sample
        # In production, you'd use the full dataset
        data_dir = 'data/openwebtext'
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if we need to download sample
        if not os.path.exists(os.path.join(data_dir, 'train.bin')):
            print("Creating synthetic OpenWebText-style sample...")
            # Create a synthetic sample that mimics web text
            sample_text = """
The quick brown fox jumps over the lazy dog. This is a sample of web text.
Machine learning has revolutionized many fields including natural language processing.
Transformers have become the dominant architecture for language modeling tasks.
The attention mechanism allows models to focus on relevant parts of the input.
""" * 1000  # Repeat to create a larger sample
            
            # Simple character encoding for demo
            chars = sorted(list(set(sample_text)))
            vocab_size = len(chars)
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
            
            # Encode and save
            data = [stoi[ch] for ch in sample_text]
            train_size = int(0.9 * len(data))
            
            train_data = np.array(data[:train_size], dtype=np.uint16)
            val_data = np.array(data[train_size:], dtype=np.uint16)
            
            train_data.tofile(os.path.join(data_dir, 'train.bin'))
            val_data.tofile(os.path.join(data_dir, 'val.bin'))
            
            # Save meta
            meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
            with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta, f)
                
            return vocab_size, 'char'
        else:
            with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
                meta = pickle.load(f)
            return meta['vocab_size'], 'char'
    
    def create_code_dataset(self):
        """Create a Python code dataset for testing."""
        print("\n📚 Creating Python code dataset...")
        data_dir = 'data/python_code'
        os.makedirs(data_dir, exist_ok=True)
        
        # Sample Python code
        code_text = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
''' * 100  # Repeat for larger dataset
        
        # Character-level encoding
        chars = sorted(list(set(code_text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode and save
        data = [stoi[ch] for ch in code_text]
        train_size = int(0.9 * len(data))
        
        train_data = np.array(data[:train_size], dtype=np.uint16)
        val_data = np.array(data[train_size:], dtype=np.uint16)
        
        train_data.tofile(os.path.join(data_dir, 'train.bin'))
        val_data.tofile(os.path.join(data_dir, 'val.bin'))
        
        # Save meta
        meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
            
        return vocab_size, 'char'
    
    def benchmark_model(self, dataset_name, data_dir, vocab_size, vocab_type, 
                       model_type='efficient', train_iters=500):
        """Benchmark a model on a specific dataset."""
        print(f"\n🔬 Benchmarking {model_type} model on {dataset_name}...")
        
        # Load data
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                              dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                            dtype=np.uint16, mode='r')
        
        # Create model config
        base_config = {
            'block_size': 256,
            'vocab_size': vocab_size,
            'n_layer': 6,
            'n_head': 6,
            'n_embd': 384,
            'dropout': 0.1,
            'bias': False
        }
        
        if model_type == 'efficient':
            efficiency = EfficiencyConfig(
                enable_early_exit=True,
                exit_threshold=0.9,
                min_exit_layer=2,
                enable_sparse_activation=True,
                sparsity_ratio=0.1,
                enable_local_attention=True,
                local_attention_window=64,
                enable_adaptive_compute=True,
                max_pondering_steps=3,
                track_efficiency_metrics=True
            )
            config = EfficientGPTConfig(**base_config, efficiency=efficiency)
            model = EfficientGPT(config).to(self.device)
        else:
            config = GPTConfig(**base_config)
            model = GPT(config).to(self.device)
        
        print(f"Model size: {model.get_num_params()/1e6:.2f}M parameters")
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        batch_size = 32
        
        # Track metrics
        train_losses = []
        val_losses = []
        start_time = time.time()
        
        # Training loop
        model.train()
        for iter in range(train_iters):
            # Get batch
            ix = torch.randint(len(train_data) - config.block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((train_data[i:i+config.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((train_data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward backward
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if iter % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_ix = torch.randint(len(val_data) - config.block_size, (batch_size,))
                    val_x = torch.stack([torch.from_numpy((val_data[i:i+config.block_size]).astype(np.int64)) for i in val_ix])
                    val_y = torch.stack([torch.from_numpy((val_data[i+1:i+1+config.block_size]).astype(np.int64)) for i in val_ix])
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    _, val_loss = model(val_x, val_y)
                    val_losses.append(val_loss.item())
                model.train()
                
                print(f"  Iter {iter}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")
        
        train_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_train_loss = np.mean(train_losses[-10:])
        final_val_loss = np.mean(val_losses[-3:]) if val_losses else final_train_loss
        
        # Measure inference speed
        tokens_processed = 0
        inference_start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                test_x = torch.randint(0, vocab_size, (8, config.block_size)).to(self.device)
                _ = model(test_x)
                tokens_processed += 8 * config.block_size
        
        inference_time = time.time() - inference_start
        tokens_per_sec = tokens_processed / inference_time
        
        # Memory usage
        if self.device == 'cuda':
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0
        
        # Get efficiency stats
        efficiency_stats = {}
        if hasattr(model, 'get_efficiency_summary'):
            efficiency_stats = model.get_efficiency_summary() or {}
        
        # Generate sample
        sample_text = self.generate_sample(model, data_dir, vocab_type)
        
        # Create result
        result = BenchmarkResult(
            dataset=dataset_name,
            model_type=model_type,
            vocab_type=vocab_type,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            train_time=train_time,
            tokens_per_sec=tokens_per_sec,
            memory_mb=memory_mb,
            efficiency_stats=efficiency_stats,
            sample_text=sample_text
        )
        
        self.results.append(result)
        return result
    
    def generate_sample(self, model, data_dir, vocab_type, max_tokens=50):
        """Generate a sample from the model."""
        # Load vocab
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        if 'stoi' in meta:
            stoi = meta['stoi']
            itos = meta['itos']
            encode = lambda s: [stoi.get(c, 0) for c in s]
            decode = lambda l: ''.join([itos.get(i, '') for i in l])
        else:
            # For BPE, use simple approach
            decode = lambda l: ''.join([str(i) for i in l])
            encode = lambda s: [0]  # Dummy
        
        # Simple prompt
        prompt_tokens = encode("\n")
        x = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)[None, ...]
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
            
        return decode(y[0].tolist())
    
    def run_comprehensive_benchmark(self):
        """Run benchmarks on all datasets."""
        print("🚀 Starting Comprehensive Benchmark")
        print("="*60)
        
        datasets = []
        
        # 1. Shakespeare (character-level) - already prepared
        datasets.append(('Shakespeare (Char)', 'data/shakespeare_char', 65, 'char'))
        
        # 2. Shakespeare (BPE)
        vocab_size, vocab_type = self.prepare_shakespeare_bpe()
        datasets.append(('Shakespeare (BPE)', 'data/shakespeare', vocab_size, vocab_type))
        
        # 3. OpenWebText sample
        vocab_size, vocab_type = self.prepare_openwebtext_sample()
        datasets.append(('Web Text', 'data/openwebtext', vocab_size, vocab_type))
        
        # 4. Python code
        vocab_size, vocab_type = self.create_code_dataset()
        datasets.append(('Python Code', 'data/python_code', vocab_size, vocab_type))
        
        # Benchmark each dataset with both models
        for dataset_name, data_dir, vocab_size, vocab_type in datasets:
            # Standard model
            self.benchmark_model(dataset_name, data_dir, vocab_size, vocab_type, 
                               model_type='standard', train_iters=300)
            
            # Efficient model
            self.benchmark_model(dataset_name, data_dir, vocab_size, vocab_type, 
                               model_type='efficient', train_iters=300)
            
            # Clear GPU cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("="*80)
        
        # Group results by dataset
        datasets = {}
        for result in self.results:
            if result.dataset not in datasets:
                datasets[result.dataset] = {}
            datasets[result.dataset][result.model_type] = result
        
        # Summary table
        print("\n### Performance Summary\n")
        print("| Dataset | Model | Train Loss | Val Loss | Tokens/sec | Memory (MB) | Speedup |")
        print("|---------|-------|------------|----------|------------|-------------|---------|")
        
        speedups = []
        
        for dataset_name in datasets:
            if 'standard' in datasets[dataset_name] and 'efficient' in datasets[dataset_name]:
                std = datasets[dataset_name]['standard']
                eff = datasets[dataset_name]['efficient']
                
                speedup = eff.tokens_per_sec / std.tokens_per_sec
                speedups.append(speedup)
                
                print(f"| {dataset_name} | Standard | {std.train_loss:.3f} | {std.val_loss:.3f} | "
                      f"{std.tokens_per_sec:.0f} | {std.memory_mb:.0f} | - |")
                print(f"| | Efficient | {eff.train_loss:.3f} | {eff.val_loss:.3f} | "
                      f"{eff.tokens_per_sec:.0f} | {eff.memory_mb:.0f} | {speedup:.2f}x |")
        
        avg_speedup = np.mean(speedups)
        print(f"\n**Average Speedup: {avg_speedup:.2f}x**")
        
        # Efficiency metrics
        print("\n### Efficiency Metrics\n")
        for dataset_name in datasets:
            if 'efficient' in datasets[dataset_name]:
                eff = datasets[dataset_name]['efficient']
                if eff.efficiency_stats:
                    print(f"\n**{dataset_name}:**")
                    for key, value in eff.efficiency_stats.items():
                        if value is not None:
                            print(f"- {key}: {value}")
        
        # Sample generations
        print("\n### Sample Generations\n")
        for dataset_name in datasets:
            print(f"\n**{dataset_name}:**")
            if 'efficient' in datasets[dataset_name]:
                sample = datasets[dataset_name]['efficient'].sample_text[:100]
                print(f"```\n{sample}...\n```")
        
        # Save detailed results
        with open('benchmark_results.json', 'w') as f:
            json.dump([{
                'dataset': r.dataset,
                'model_type': r.model_type,
                'train_loss': r.train_loss,
                'val_loss': r.val_loss,
                'tokens_per_sec': r.tokens_per_sec,
                'memory_mb': r.memory_mb,
                'efficiency_stats': r.efficiency_stats
            } for r in self.results], f, indent=2)
        
        print("\n✅ Detailed results saved to benchmark_results.json")
        
        return avg_speedup, datasets


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    benchmark = DatasetBenchmark(device)
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()