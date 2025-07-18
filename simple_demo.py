"""
Simple demo script to test the Efficient GPT model.
You can modify the prompts below to test different inputs!
"""

import os
import torch
import pickle
import numpy as np
import time

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


def demo_efficient_gpt():
    """Demo the efficient GPT with various prompts."""
    
    # Setup
    os.chdir('/home/ruo/my_project/my-efficient-gpt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load vocabulary
    data_dir = 'data/shakespeare_char'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    vocab_size = meta['vocab_size']
    
    encode = lambda s: [stoi.get(ch, stoi.get(' ', 0)) for ch in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    # Create efficient model with all features
    print("\n🏗️  Creating Efficient GPT model...")
    config = EfficientGPTConfig(
        block_size=256,
        vocab_size=vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        efficiency=EfficiencyConfig(
            enable_early_exit=True,
            exit_threshold=0.85,
            min_exit_layer=2,
            enable_sparse_activation=True,
            sparsity_ratio=0.1,  # 90% sparse
            enable_local_attention=True,
            local_attention_window=64,
            enable_adaptive_compute=True,
            max_pondering_steps=3,
            track_efficiency_metrics=True
        )
    )
    
    model = EfficientGPT(config).to(device)
    print(f"✅ Model created with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Quick training
    print("\n🎓 Quick training (500 steps)...")
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    batch_size = 32
    block_size = config.block_size
    
    model.train()
    for i in range(500):
        # Get batch
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((train_data[j:j+block_size]).astype(np.int64)) for j in ix])
        y = torch.stack([torch.from_numpy((train_data[j+1:j+1+block_size]).astype(np.int64)) for j in ix])
        x, y = x.to(device), y.to(device)
        
        # Forward backward
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"  Step {i}: loss {loss.item():.4f}")
    
    print("✅ Training complete!")
    
    # Test generation with different prompts
    print("\n" + "="*60)
    print("🎭 TESTING SHAKESPEARE GENERATION WITH EFFICIENCY FEATURES")
    print("="*60)
    
    # YOU CAN MODIFY THESE PROMPTS!
    test_prompts = [
        "To be or not to be",
        "ROMEO:\nI love",
        "All the world's a stage,\nAnd all the men and women",
        "O ",
        "HAMLET:\nTo die, to sleep",
        "Friends, Romans, countrymen,",
        "If music be the food of love,",
        "What's in a name? That which we call",
    ]
    
    model.eval()
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"📝 PROMPT: '{prompt}'")
        print("="*60)
        
        # Encode and generate
        tokens = encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
        
        # Test with different temperatures
        for temp in [0.6, 0.8, 1.0]:
            print(f"\n🌡️  Temperature = {temp}:")
            
            start_time = time.time()
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=80, temperature=temp, top_k=40)
            gen_time = time.time() - start_time
            
            generated = decode(y[0].tolist())
            print(f"   {generated}")
            
            # Show efficiency stats
            if hasattr(model, 'get_efficiency_summary'):
                stats = model.get_efficiency_summary()
                if stats:
                    print(f"\n   📊 Efficiency stats:")
                    if stats.get('avg_exit_layer'):
                        print(f"      Early exit: layer {stats['avg_exit_layer']:.1f}/{config.n_layer}")
                    if stats.get('avg_sparsity'):
                        print(f"      Sparsity: {stats['avg_sparsity']:.1%}")
                    if stats.get('avg_ponder_steps'):
                        print(f"      Pondering: {stats['avg_ponder_steps']:.1f} steps")
                    print(f"      Speed: {80/gen_time:.1f} tokens/sec")
    
    # Compare with efficiency features disabled
    print("\n" + "="*60)
    print("🔄 COMPARING WITH EFFICIENCY FEATURES DISABLED")
    print("="*60)
    
    # Disable features
    model.efficiency_config.enable_early_exit = False
    model.efficiency_config.enable_sparse_activation = False
    
    prompt = "To be or not to be, that is the question"
    tokens = encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    print(f"\nPrompt: '{prompt}'")
    
    # Generate with features disabled
    start_time = time.time()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
    gen_time_disabled = time.time() - start_time
    
    print(f"\nWithout efficiency features:")
    print(f"  Generated in: {gen_time_disabled:.2f}s")
    print(f"  Speed: {100/gen_time_disabled:.1f} tokens/sec")
    
    # Re-enable features
    model.efficiency_config.enable_early_exit = True
    model.efficiency_config.enable_sparse_activation = True
    
    start_time = time.time()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
    gen_time_enabled = time.time() - start_time
    
    print(f"\nWith efficiency features:")
    print(f"  Generated in: {gen_time_enabled:.2f}s")
    print(f"  Speed: {100/gen_time_enabled:.1f} tokens/sec")
    print(f"  Speedup: {gen_time_disabled/gen_time_enabled:.2f}x")
    
    print("\n✅ Demo complete! Feel free to modify the prompts in the script and run again!")


if __name__ == "__main__":
    demo_efficient_gpt()