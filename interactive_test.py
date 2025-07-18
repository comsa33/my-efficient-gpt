"""
Interactive testing script for Efficient GPT.
You can chat with the model and see efficiency features in action!
"""

import os
import torch
import pickle
import numpy as np
from contextlib import nullcontext
import time

from model import GPT, GPTConfig
from model_efficient import EfficientGPT, EfficientGPTConfig, EfficiencyConfig


class InteractiveGPT:
    def __init__(self, model_type='efficient', checkpoint_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
        # Load vocabulary
        self.data_dir = 'data/shakespeare_char'
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = meta['vocab_size']
        
        # Create or load model
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            self.create_model(model_type)
            if not checkpoint_path:
                self.quick_train()
    
    def create_model(self, model_type):
        """Create a new model."""
        print(f"\n🏗️  Creating {model_type} model...")
        
        base_config = {
            'block_size': 256,
            'vocab_size': self.vocab_size,
            'n_layer': 6,
            'n_head': 6,
            'n_embd': 384,
            'dropout': 0.0,
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
            self.model = EfficientGPT(config).to(self.device)
        else:
            config = GPTConfig(**base_config)
            self.model = GPT(config).to(self.device)
        
        print(f"✅ Model created with {self.model.get_num_params()/1e6:.2f}M parameters")
    
    def quick_train(self, iterations=500):
        """Quick training to get reasonable outputs."""
        print(f"\n🎓 Quick training for {iterations} iterations...")
        
        # Load training data
        train_data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        
        # Training loop
        batch_size = 32
        block_size = self.model.config.block_size
        
        self.model.train()
        for i in range(iterations):
            # Get batch
            ix = torch.randint(len(train_data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((train_data[j:j+block_size]).astype(np.int64)) for j in ix])
            y = torch.stack([torch.from_numpy((train_data[j+1:j+1+block_size]).astype(np.int64)) for j in ix])
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward backward
            logits, loss = self.model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"  Step {i}: loss {loss.item():.4f}")
        
        print("✅ Training complete!")
    
    def encode(self, text):
        """Encode text to tokens."""
        return [self.stoi.get(ch, self.stoi.get(' ', 0)) for ch in text]
    
    def decode(self, tokens):
        """Decode tokens to text."""
        return ''.join([self.itos.get(i, '?') for i in tokens])
    
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=40):
        """Generate text from prompt."""
        self.model.eval()
        
        # Encode prompt
        tokens = self.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            y = self.model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        gen_time = time.time() - start_time
        
        # Decode
        generated = self.decode(y[0].tolist())
        
        # Get efficiency stats
        stats = {}
        if hasattr(self.model, 'get_efficiency_summary'):
            stats = self.model.get_efficiency_summary()
        
        return generated, gen_time, stats
    
    def interactive_mode(self):
        """Interactive chat with the model."""
        print("\n" + "="*60)
        print("🎭 INTERACTIVE SHAKESPEARE GPT")
        print("="*60)
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /stats    - Show efficiency statistics")
        print("  /temp X   - Set temperature (0.1-2.0)")
        print("  /tokens X - Set max tokens (10-500)")
        print("  /mode     - Toggle efficiency features")
        print("  /quit     - Exit")
        print("\nOr just type text to generate Shakespeare-style continuations!")
        print("="*60)
        
        temperature = 0.8
        max_tokens = 100
        
        while True:
            try:
                prompt = input("\n📝 Enter text (or command): ").strip()
                
                if not prompt:
                    continue
                
                # Handle commands
                if prompt.startswith('/'):
                    parts = prompt.split()
                    cmd = parts[0].lower()
                    
                    if cmd == '/quit':
                        print("👋 Goodbye!")
                        break
                    
                    elif cmd == '/help':
                        print("\nCommands:")
                        print("  /help     - Show this help")
                        print("  /stats    - Show efficiency statistics")
                        print("  /temp X   - Set temperature (0.1-2.0)")
                        print("  /tokens X - Set max tokens (10-500)")
                        print("  /mode     - Toggle efficiency features")
                        print("  /quit     - Exit")
                    
                    elif cmd == '/stats':
                        if hasattr(self.model, 'metrics') and self.model.metrics:
                            summary = self.model.metrics.get_summary()
                            print("\n📊 Efficiency Statistics:")
                            for key, value in summary.items():
                                if value is not None:
                                    print(f"  {key}: {value}")
                        else:
                            print("No efficiency statistics available")
                    
                    elif cmd == '/temp' and len(parts) > 1:
                        try:
                            temperature = float(parts[1])
                            temperature = max(0.1, min(2.0, temperature))
                            print(f"✅ Temperature set to {temperature}")
                        except:
                            print("❌ Invalid temperature. Use a number between 0.1 and 2.0")
                    
                    elif cmd == '/tokens' and len(parts) > 1:
                        try:
                            max_tokens = int(parts[1])
                            max_tokens = max(10, min(500, max_tokens))
                            print(f"✅ Max tokens set to {max_tokens}")
                        except:
                            print("❌ Invalid token count. Use a number between 10 and 500")
                    
                    elif cmd == '/mode':
                        if hasattr(self.model, 'efficiency_config'):
                            # Toggle efficiency features
                            cfg = self.model.efficiency_config
                            cfg.enable_early_exit = not cfg.enable_early_exit
                            cfg.enable_sparse_activation = not cfg.enable_sparse_activation
                            print(f"✅ Efficiency features toggled:")
                            print(f"  Early exit: {cfg.enable_early_exit}")
                            print(f"  Sparse activation: {cfg.enable_sparse_activation}")
                        else:
                            print("❌ This model doesn't have efficiency features")
                    
                    else:
                        print("❌ Unknown command. Type /help for commands.")
                
                else:
                    # Generate text
                    print(f"\n🤖 Generating (temp={temperature}, max_tokens={max_tokens})...")
                    
                    generated, gen_time, stats = self.generate(
                        prompt, 
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    print("\n" + "-"*60)
                    print(generated)
                    print("-"*60)
                    
                    # Show stats
                    print(f"\n⏱️  Generated in {gen_time:.2f}s ({max_tokens/gen_time:.1f} tokens/sec)")
                    
                    if stats:
                        print("📊 Efficiency stats for this generation:")
                        if stats.get('avg_exit_layer'):
                            print(f"  - Early exit at layer {stats['avg_exit_layer']:.1f}/{self.model.config.n_layer}")
                        if stats.get('avg_sparsity'):
                            print(f"  - Sparsity: {stats['avg_sparsity']:.1%}")
                        if stats.get('avg_ponder_steps'):
                            print(f"  - Pondering steps: {stats['avg_ponder_steps']:.1f}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    print("🧠 EFFICIENT GPT INTERACTIVE TESTER")
    print("="*60)
    
    # Change to project root
    os.chdir('/home/ruo/my_project/my-efficient-gpt')
    
    # Check if user wants to use existing checkpoint
    print("\nOptions:")
    print("1. Create and train a new model (quick)")
    print("2. Use existing checkpoint (if available)")
    
    choice = input("\nYour choice (1 or 2): ").strip()
    
    if choice == '2':
        checkpoint_path = input("Enter checkpoint path (or press Enter to skip): ").strip()
        if not checkpoint_path:
            checkpoint_path = None
    else:
        checkpoint_path = None
    
    # Model type
    print("\nModel type:")
    print("1. Efficient GPT (with brain-inspired features)")
    print("2. Standard GPT (baseline)")
    
    model_choice = input("\nYour choice (1 or 2): ").strip()
    model_type = 'efficient' if model_choice != '2' else 'standard'
    
    # Create tester
    tester = InteractiveGPT(model_type, checkpoint_path)
    
    # Run interactive mode
    tester.interactive_mode()


if __name__ == "__main__":
    main()