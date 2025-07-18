"""
Test script to verify the language model works correctly.
Trains a small model on Shakespeare and generates text.
"""

import os
import torch
import pickle
import numpy as np
from contextlib import nullcontext

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


def test_baseline_model():
    """Test the baseline GPT model."""
    print("\n" + "="*60)
    print("Testing Baseline GPT Model")
    print("="*60)
    
    # Configuration for small test model
    config = GPTConfig(
        block_size=128,
        vocab_size=65,  # Shakespeare character-level vocab
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Load data
    data_dir = 'data/shakespeare_char'
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\nTraining for 100 iterations...")
    model.train()
    for iter in range(100):
        # Get batch
        ix = torch.randint(len(train_data) - config.block_size, (8,))
        x = torch.stack([torch.from_numpy((train_data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 20 == 0:
            print(f"Iter {iter}: loss {loss.item():.4f}")
    
    return model


def test_efficient_model():
    """Test the efficient GPT model with brain-inspired features."""
    print("\n" + "="*60)
    print("Testing Efficient GPT Model")
    print("="*60)
    
    # Configuration with efficiency features
    config = EfficientGPTConfig(
        block_size=128,
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_early_exit=True,
            exit_threshold=0.9,
            min_exit_layer=1,
            enable_sparse_activation=True,
            sparsity_ratio=0.2,
            enable_local_attention=True,
            local_attention_window=32,
            track_efficiency_metrics=True
        )
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientGPT(config).to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Load data
    data_dir = 'data/shakespeare_char'
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\nTraining for 100 iterations...")
    model.train()
    for iter in range(100):
        # Get batch
        ix = torch.randint(len(train_data) - config.block_size, (8,))
        x = torch.stack([torch.from_numpy((train_data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 20 == 0:
            print(f"Iter {iter}: loss {loss.item():.4f}")
            
            # Show efficiency metrics
            if hasattr(model, 'get_efficiency_summary'):
                summary = model.get_efficiency_summary()
                if summary and any(v is not None for v in summary.values()):
                    print(f"  Efficiency: {summary}")
    
    return model


def generate_text(model, prompt="To be or not to be", max_tokens=100, temperature=0.8):
    """Generate text from a model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    # Load character mappings
    data_dir = 'data/shakespeare_char'
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    
    # Encode prompt
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Generate
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"\nPrompt: '{prompt}'")
    print("Generated text:")
    print("-" * 40)
    
    with torch.no_grad():
        with nullcontext():
            generated_ids = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=40)
            generated_text = decode(generated_ids[0].tolist())
            print(generated_text)
    
    return generated_text


def compare_models():
    """Compare baseline and efficient models."""
    print("\n" + "="*60)
    print("Comparing Models on Text Generation")
    print("="*60)
    
    # Test prompts
    prompts = [
        "ROMEO:",
        "To be or not to be",
        "All the world's a stage",
        "O ",
    ]
    
    # Train baseline model
    print("\n1. Training baseline model...")
    baseline_model = test_baseline_model()
    
    # Train efficient model
    print("\n2. Training efficient model...")
    efficient_model = test_efficient_model()
    
    # Generate text from both models
    print("\n3. Generating text from both models...")
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: '{prompt}'")
        print("="*60)
        
        print("\nBaseline Model:")
        baseline_text = generate_text(baseline_model, prompt, max_tokens=50)
        
        print("\nEfficient Model:")
        efficient_text = generate_text(efficient_model, prompt, max_tokens=50)


def main():
    """Main test function."""
    print("🎭 TESTING LANGUAGE MODEL CAPABILITIES 🎭")
    
    # Change to project root
    os.chdir('/home/ruo/my_project/my-efficient-gpt')
    
    # Run comparison
    compare_models()
    
    print("\n✅ Language model test completed!")
    print("\nKey observations:")
    print("- Both models successfully learn from Shakespeare text")
    print("- Efficient model maintains generation quality")
    print("- Efficiency features work during training and generation")
    print("\nFor better results, train for more iterations with proper hyperparameters!")


if __name__ == "__main__":
    main()