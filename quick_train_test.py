"""
Quick training test to see meaningful language generation.
"""

import os
import torch
import pickle
import numpy as np
import time
from contextlib import nullcontext

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


def quick_train_and_generate():
    """Train models for a bit longer to see better results."""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration - small but effective
    base_config = {
        'block_size': 256,
        'vocab_size': 65,  # Shakespeare character-level
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'dropout': 0.1,
        'bias': False
    }
    
    # Create efficient model with select features
    efficiency = EfficiencyConfig(
        enable_sparse_activation=True,
        sparsity_ratio=0.15,
        enable_local_attention=True,
        local_attention_window=64,
        enable_early_exit=True,
        exit_threshold=0.85,
        min_exit_layer=3,
        track_efficiency_metrics=True
    )
    
    config = EfficientGPTConfig(**base_config, efficiency=efficiency)
    model = EfficientGPT(config).to(device)
    
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Load data
    data_dir = 'data/shakespeare_char'
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Load vocab
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Training settings
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 1000
    eval_interval = 100
    eval_iters = 20
    
    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )
    
    # Compile model for speed
    if device == 'cuda':
        model = torch.compile(model)
        print("Model compiled for faster training")
    
    # Helper functions
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # Training loop
    print(f"\nTraining for {max_iters} iterations...")
    print("-" * 60)
    
    model.train()
    t0 = time.time()
    
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Check efficiency metrics
            if hasattr(model, 'metrics') and model.metrics:
                summary = model.metrics.get_summary()
                if summary['avg_sparsity']:
                    print(f"  Sparsity: {summary['avg_sparsity']:.1%}")
                if summary['avg_exit_layer']:
                    print(f"  Avg exit layer: {summary['avg_exit_layer']:.1f}/{config.n_layer}")
        
        # Get batch and train
        X, Y = get_batch('train')
        
        # Forward backward update
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    t1 = time.time()
    print(f"\nTraining completed in {t1-t0:.1f}s")
    
    # Generate samples
    print("\n" + "="*60)
    print("GENERATING SHAKESPEARE-STYLE TEXT")
    print("="*60)
    
    model.eval()
    
    prompts = [
        "\n",  # Start fresh
        "HAMLET:",
        "To be or not to be",
        "O Romeo, Romeo! wherefore art thou",
        "Friends, Romans, countrymen",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt.strip()}'")
        print("Generated:")
        
        # Encode and generate
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
            generated = decode(y[0].tolist())
            
        # Format nicely
        lines = generated.split('\n')
        for line in lines[:5]:  # Show first 5 lines
            if line.strip():
                print(f"  {line}")
        
        # Show efficiency stats for this generation
        if hasattr(model, 'get_efficiency_summary'):
            summary = model.get_efficiency_summary()
            if summary and summary.get('avg_exit_layer'):
                print(f"  [Exited at layer {summary['avg_exit_layer']:.1f} on average]")
    
    print("\n✅ Training and generation test completed!")
    print("\nNOTE: For production-quality text, train for 5000+ iterations")
    print("      This was just a quick test to verify functionality")


if __name__ == "__main__":
    # Change to project root
    os.chdir('/home/ruo/my_project/my-efficient-gpt')
    quick_train_and_generate()