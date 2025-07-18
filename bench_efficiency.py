"""
Benchmarking script for comparing standard GPT vs Efficient GPT.
Measures performance, memory usage, and efficiency metrics.
"""

import os
import time
import torch
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    config_name: str
    batch_size: int
    seq_length: int
    
    # Timing
    forward_time: float
    backward_time: float
    total_time: float
    tokens_per_second: float
    
    # Memory
    peak_memory_mb: float
    allocated_memory_mb: float
    
    # Model metrics
    mfu: float
    num_params: int
    
    # Efficiency metrics (if available)
    efficiency_metrics: Dict = None


@contextmanager
def track_memory():
    """Context manager to track GPU memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()


def benchmark_model(model, batch_size, seq_length, device, num_iterations=10):
    """Benchmark a model's forward and backward pass."""
    model.train()
    
    # Prepare input
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
    
    # Warmup
    for _ in range(3):
        logits, loss = model(x, y)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Benchmark
    forward_times = []
    backward_times = []
    
    with track_memory():
        for _ in range(num_iterations):
            # Forward pass
            torch.cuda.synchronize()
            t0 = time.time()
            logits, loss = model(x, y)
            torch.cuda.synchronize()
            t1 = time.time()
            forward_times.append(t1 - t0)
            
            # Backward pass
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.time()
            backward_times.append(t1 - t0)
    
    # Calculate metrics
    forward_time = np.mean(forward_times)
    backward_time = np.mean(backward_times)
    total_time = forward_time + backward_time
    
    tokens_processed = batch_size * seq_length
    tokens_per_second = tokens_processed / total_time
    
    # Memory metrics
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # MFU estimation
    mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=total_time)
    
    # Get efficiency metrics if available
    efficiency_metrics = None
    if hasattr(model, 'get_efficiency_summary'):
        efficiency_metrics = model.get_efficiency_summary()
    
    return BenchmarkResult(
        model_name=model.__class__.__name__,
        config_name=f"L{model.config.n_layer}_H{model.config.n_head}_D{model.config.n_embd}",
        batch_size=batch_size,
        seq_length=seq_length,
        forward_time=forward_time,
        backward_time=backward_time,
        total_time=total_time,
        tokens_per_second=tokens_per_second,
        peak_memory_mb=peak_memory,
        allocated_memory_mb=allocated_memory,
        mfu=mfu,
        num_params=model.get_num_params(),
        efficiency_metrics=efficiency_metrics
    )


def create_model_configs():
    """Create different model configurations for testing."""
    configs = []
    
    # Small model for testing
    base_config = GPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False
    )
    configs.append(("base_small", base_config))
    
    # Efficient model with all features
    eff_config_full = EfficientGPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_early_exit=True,
            exit_threshold=0.95,
            min_exit_layer=2,
            enable_sparse_activation=True,
            sparsity_ratio=0.1,
            enable_adaptive_compute=True,
            max_pondering_steps=3,
            enable_dynamic_routing=True,
            enable_local_attention=True,
            local_attention_window=128
        )
    )
    configs.append(("efficient_full", eff_config_full))
    
    # Efficient model with only early exit
    eff_config_exit = EfficientGPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_early_exit=True,
            exit_threshold=0.95,
            min_exit_layer=2
        )
    )
    configs.append(("efficient_exit", eff_config_exit))
    
    # Efficient model with only sparse activation
    eff_config_sparse = EfficientGPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_sparse_activation=True,
            sparsity_ratio=0.1
        )
    )
    configs.append(("efficient_sparse", eff_config_sparse))
    
    # Efficient model with local attention
    eff_config_local = EfficientGPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_local_attention=True,
            local_attention_window=128
        )
    )
    configs.append(("efficient_local", eff_config_local))
    
    return configs


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*120)
    print("BENCHMARK RESULTS")
    print("="*120)
    
    # Basic metrics
    print(f"\n{'Model':<20} {'Config':<15} {'Batch':<6} {'Seq':<6} {'Time(ms)':<10} {'Tokens/s':<12} {'Memory(MB)':<12} {'MFU':<8}")
    print("-"*120)
    
    base_time = None
    base_memory = None
    
    for r in results:
        if r.model_name == "GPT":
            base_time = r.total_time
            base_memory = r.peak_memory_mb
        
        time_str = f"{r.total_time*1000:.2f}"
        if base_time and r.model_name != "GPT":
            speedup = base_time / r.total_time
            time_str += f" ({speedup:.2f}x)"
        
        mem_str = f"{r.peak_memory_mb:.1f}"
        if base_memory and r.model_name != "GPT":
            mem_ratio = r.peak_memory_mb / base_memory
            mem_str += f" ({mem_ratio:.2f}x)"
        
        print(f"{r.config_name:<20} {r.config_name:<15} {r.batch_size:<6} {r.seq_length:<6} "
              f"{time_str:<10} {r.tokens_per_second:<12.0f} {mem_str:<12} {r.mfu:<8.4f}")
    
    # Efficiency metrics
    print(f"\n{'='*120}")
    print("EFFICIENCY METRICS")
    print("="*120)
    
    for r in results:
        if r.efficiency_metrics:
            print(f"\n{r.config_name}:")
            for key, value in r.efficiency_metrics.items():
                if value is not None:
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU, results may not be representative")
    
    torch.manual_seed(1337)
    torch.backends.cudnn.benchmark = True
    
    # Create configurations
    configs = create_model_configs()
    
    # Benchmark settings
    batch_sizes = [4, 8]
    seq_lengths = [256, 512]
    
    results = []
    
    print("Starting benchmarks...")
    print(f"Device: {device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    
    for config_name, config in configs:
        print(f"\nBenchmarking {config_name}...")
        
        # Create model
        if isinstance(config, EfficientGPTConfig):
            model = EfficientGPT(config).to(device)
        else:
            model = GPT(config).to(device)
        
        # Compile model for better performance
        try:
            model = torch.compile(model)
        except:
            print("Could not compile model, running eager mode")
        
        # Run benchmarks for different batch sizes and sequence lengths
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                if seq_length > config.block_size:
                    continue
                
                print(f"  Batch size: {batch_size}, Seq length: {seq_length}")
                
                try:
                    result = benchmark_model(model, batch_size, seq_length, device)
                    result.config_name = config_name
                    results.append(result)
                except Exception as e:
                    print(f"    Failed: {e}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Print results
    print_results(results)
    
    # Save detailed results
    print("\nSaving detailed results to benchmark_results.txt...")
    with open("benchmark_results.txt", "w") as f:
        for r in results:
            f.write(f"{r}\n")


if __name__ == "__main__":
    main()