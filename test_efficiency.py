"""
Test script to demonstrate efficiency improvements in nanoGPT.
Tests various brain-inspired optimizations and measures their impact.
"""

import torch
import time
import os
from contextlib import contextmanager

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


@contextmanager
def timer(name):
    """Simple timer context manager."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"{name}: {time.time() - start:.3f}s")


def create_test_models(device='cuda'):
    """Create standard and efficient models for comparison."""
    # Base configuration
    base_config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False
    )
    
    # Efficient configurations
    configs = {
        'baseline': base_config,
        'early_exit': EfficientGPTConfig(
            **base_config.__dict__,
            efficiency=EfficiencyConfig(
                enable_early_exit=True,
                exit_threshold=0.9,
                min_exit_layer=2
            )
        ),
        'sparse': EfficientGPTConfig(
            **base_config.__dict__,
            efficiency=EfficiencyConfig(
                enable_sparse_activation=True,
                sparsity_ratio=0.1
            )
        ),
        'local_attention': EfficientGPTConfig(
            **base_config.__dict__,
            efficiency=EfficiencyConfig(
                enable_local_attention=True,
                local_attention_window=64
            )
        ),
        'full_efficient': EfficientGPTConfig(
            **base_config.__dict__,
            efficiency=EfficiencyConfig(
                enable_early_exit=True,
                exit_threshold=0.9,
                min_exit_layer=2,
                enable_sparse_activation=True,
                sparsity_ratio=0.1,
                enable_adaptive_compute=True,
                max_pondering_steps=2,
                enable_local_attention=True,
                local_attention_window=64,
                enable_hierarchical=True,
                hierarchical_type='multi_scale',
                track_efficiency_metrics=True
            )
        )
    }
    
    models = {}
    for name, config in configs.items():
        if name == 'baseline':
            model = GPT(config)
        else:
            model = EfficientGPT(config)
        model = model.to(device)
        model.eval()
        models[name] = model
        print(f"Created {name} model with {model.get_num_params()/1e6:.2f}M parameters")
    
    return models


def test_inference_speed(models, device='cuda', batch_size=8, seq_length=256, n_iterations=10):
    """Test inference speed of different models."""
    print(f"\n{'='*60}")
    print(f"Testing inference speed (batch_size={batch_size}, seq_length={seq_length})")
    print(f"{'='*60}")
    
    # Prepare input
    x = torch.randint(0, 50304, (batch_size, seq_length), device=device)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        for _ in range(n_iterations):
            with torch.no_grad():
                logits, _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        avg_time = elapsed / n_iterations
        tokens_per_second = (batch_size * seq_length) / avg_time
        
        results[name] = {
            'avg_time': avg_time,
            'tokens_per_second': tokens_per_second
        }
        
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Tokens/second: {tokens_per_second:,.0f}")
        
        # Print efficiency metrics if available
        if hasattr(model, 'get_efficiency_summary'):
            summary = model.get_efficiency_summary()
            if summary and any(v is not None for v in summary.values()):
                print(f"  Efficiency metrics:")
                for key, value in summary.items():
                    if value is not None:
                        print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("Performance comparison vs baseline:")
    print(f"{'='*60}")
    
    baseline_time = results['baseline']['avg_time']
    baseline_tokens = results['baseline']['tokens_per_second']
    
    for name, metrics in results.items():
        if name != 'baseline':
            speedup = baseline_time / metrics['avg_time']
            token_ratio = metrics['tokens_per_second'] / baseline_tokens
            print(f"{name}: {speedup:.2f}x speedup, {token_ratio:.2f}x tokens/sec")


def test_memory_usage(models, device='cuda', batch_size=8, seq_length=256):
    """Test memory usage of different models."""
    print(f"\n{'='*60}")
    print(f"Testing memory usage (batch_size={batch_size}, seq_length={seq_length})")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory test")
        return
    
    x = torch.randint(0, 50304, (batch_size, seq_length), device=device)
    
    results = {}
    
    for name, model in models.items():
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Measure memory
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        results[name] = {
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak
        }
        
        print(f"\n{name}:")
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved: {reserved:.1f} MB")
        print(f"  Peak: {peak:.1f} MB")
    
    # Compare memory usage
    print(f"\n{'='*60}")
    print("Memory comparison vs baseline:")
    print(f"{'='*60}")
    
    baseline_peak = results['baseline']['peak']
    
    for name, metrics in results.items():
        if name != 'baseline':
            memory_ratio = metrics['peak'] / baseline_peak
            saved = baseline_peak - metrics['peak']
            print(f"{name}: {memory_ratio:.2f}x memory ({saved:.1f} MB saved)")


def test_generation_quality(models, device='cuda', prompt="The brain is", max_tokens=50):
    """Test generation quality of different models."""
    print(f"\n{'='*60}")
    print(f"Testing generation quality")
    print(f"{'='*60}")
    
    # Simple tokenization (character-level for testing)
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\n")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode prompt (with fallback for unknown characters)
    encoded = []
    for ch in prompt:
        if ch in stoi:
            encoded.append(stoi[ch])
        else:
            encoded.append(stoi[' '])  # Use space for unknown characters
    
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Prompt: '{prompt}'")
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        with torch.no_grad():
            # Generate
            start_time = time.time()
            generated = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
            gen_time = time.time() - start_time
            
            # Decode
            generated_text = ''.join([itos.get(i.item(), '?') for i in generated[0]])
            
            print(f"  Generated: '{generated_text}'")
            print(f"  Time: {gen_time:.2f}s ({max_tokens/gen_time:.1f} tokens/sec)")


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create models
    print("\nCreating models...")
    models = create_test_models(device)
    
    # Run tests
    test_inference_speed(models, device)
    test_memory_usage(models, device)
    test_generation_quality(models, device)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()