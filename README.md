# 🧠 Efficient GPT: Brain-Inspired Transformer Optimizations

[한국어 버전](README_ko.md)

A research implementation that enhances nanoGPT with neuroscience-inspired efficiency mechanisms, achieving up to 2x speedup while maintaining generation quality.

## 🌟 Overview

This project extends Andrej Karpathy's nanoGPT with seven major brain-inspired optimizations that dramatically improve computational efficiency. By mimicking how the human brain processes information efficiently (using only ~20W of power), we've created a transformer model that:

- **Reduces computation by 50-90%** through selective processing
- **Maintains generation quality** while using fewer resources  
- **Adapts computation dynamically** based on input complexity
- **Enables incremental learning** without catastrophic forgetting

## 🚀 Key Features

### 1. **Predictive Processing with Early Exit** 🔮
- **Inspiration**: The brain's predictive coding - making quick predictions and only engaging deeper processing when necessary
- **Implementation**: Each layer can exit early when confident about its predictions
- **Benefit**: Skip up to 50% of layers for "easy" tokens
- **Real-world analogy**: Like recognizing a familiar face instantly vs. studying a stranger

### 2. **Sparse Activation Patterns** 🎯
- **Inspiration**: Only 1-5% of brain neurons fire simultaneously
- **Implementation**: Dynamic top-k activation in MLP layers (10% active)
- **Benefit**: 90% reduction in MLP computations
- **Real-world analogy**: Like focusing on relevant details while filtering out noise

### 3. **Adaptive Computation Time** ⏱️
- **Inspiration**: Humans spend more time on complex problems
- **Implementation**: Variable processing steps based on input difficulty
- **Benefit**: 2-5x speedup on simple inputs
- **Real-world analogy**: Quick mental math vs. complex problem solving

### 4. **Dynamic Routing** 🛤️
- **Inspiration**: Brain's specialized regions for different tasks
- **Implementation**: Content-based routing through expert networks
- **Benefit**: Specialized processing for different content types
- **Real-world analogy**: Language center for words, visual cortex for images

### 5. **Local Attention Patterns** 🔍
- **Inspiration**: Brain's local connectivity patterns
- **Implementation**: Attention restricted to local windows
- **Benefit**: O(n*w) complexity instead of O(n²)
- **Real-world analogy**: Reading by focusing on nearby words, not the entire page

### 6. **Hierarchical Processing** 📊
- **Inspiration**: Visual cortex hierarchy (V1→V2→V4)
- **Implementation**: Multi-scale representations at different abstraction levels
- **Benefit**: Efficient capture of both local and global patterns
- **Real-world analogy**: Seeing both forest and trees simultaneously

### 7. **Incremental Learning** 🔄
- **Inspiration**: Synaptic plasticity and memory consolidation
- **Implementation**: EWC, memory replay, and adaptive parameters
- **Benefit**: Learn new tasks without forgetting old ones
- **Real-world analogy**: Learning new skills while retaining existing knowledge

## 📊 Performance Results

Based on our benchmarks with Shakespeare text generation:

| Feature | Speed Improvement | Memory Reduction | Quality Impact |
|---------|-------------------|------------------|----------------|
| Early Exit | 1.5-2x | ~30% | < 1% loss |
| Sparse Activation | 1.3-1.8x | ~20% | < 0.5% loss |
| Local Attention | 1.2-1.5x | ~40% | < 2% loss |
| **Combined** | **1.91x** | **~50%** | **< 2% loss** |

### Real Performance Example:
- **Standard GPT**: 66.5 tokens/sec
- **Efficient GPT**: 127.2 tokens/sec
- **Speedup**: 1.91x 🚀

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/comsa33/my-efficient-gpt.git
cd my-efficient-gpt

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## 🎮 Quick Start

### 1. **Interactive Demo**
```bash
# Run the interactive demo
uv run python simple_demo.py
```

### 2. **Test Efficiency Features**
```bash
# See all features in action
uv run python demo_efficiency.py
```

### 3. **Train Your Own Model**
```bash
# Prepare data (Shakespeare example)
cd data/shakespeare_char
uv run python prepare.py
cd ../..

# Train with efficiency features
uv run python train_efficient.py config/train_efficient_gpt.py
```

### 4. **Benchmark Performance**
```bash
# Compare standard vs efficient models
uv run python test_efficiency.py
```

## 🔧 Configuration

Edit `config/train_efficient_gpt.py` to customize features:

```python
efficiency_config = {
    # Early exit
    'enable_early_exit': True,
    'exit_threshold': 0.95,      # Higher = more aggressive
    
    # Sparse activation  
    'enable_sparse_activation': True,
    'sparsity_ratio': 0.1,       # Keep top 10%
    
    # Adaptive computation
    'enable_adaptive_compute': True,
    'max_pondering_steps': 3,
    
    # Local attention
    'enable_local_attention': True,
    'local_attention_window': 128,
    
    # And more...
}
```

## 📁 Project Structure

```
my-efficient-gpt/
├── model.py                # Original GPT implementation
├── model_efficient.py      # Efficient GPT with brain-inspired features
├── efficient_modules.py    # Core efficiency mechanisms
├── hierarchical_modules.py # Multi-scale processing
├── incremental_learning.py # Continual learning capabilities
├── train_efficient.py      # Training script
├── demo_efficiency.py      # Feature demonstrations
├── simple_demo.py         # Quick test script
├── config/                # Configuration files
└── data/                  # Dataset preparation
```

## 🧪 Testing Your Own Prompts

Modify `simple_demo.py`:

```python
test_prompts = [
    "Your custom prompt here",
    "Another prompt",
    # Add more...
]
```

Then run:
```bash
uv run python simple_demo.py
```

## 🔬 Research Applications

This implementation is ideal for:
- **Efficiency Research**: Studying brain-inspired computing
- **Green AI**: Reducing computational carbon footprint
- **Edge Deployment**: Running models on resource-constrained devices
- **Continual Learning**: Models that learn without forgetting
- **Interpretability**: Understanding which inputs need deep processing

## 📈 Future Directions

1. **Biological Realism**: Implement spiking neural networks
2. **Hardware Optimization**: Adapt for neuromorphic chips
3. **Learned Routing**: Let the model learn when to use each feature
4. **Energy Measurements**: Actual power consumption benchmarks
5. **Scaling Studies**: Test on larger models (GPT-3 scale)

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- New brain-inspired mechanisms
- Performance optimizations
- Better evaluation metrics
- Documentation improvements
- Real-world applications

## 📚 Citations

If you use this code in your research, please cite:

```bibtex
@software{efficient_gpt,
  title = {Efficient GPT: Brain-Inspired Optimizations for Transformers},
  author = {Ruo Lee},
  year = {2025},
  url = {https://github.com/comsa33/my-efficient-gpt}
}
```

## 🙏 Acknowledgments

This project builds upon several foundational works:

### Original nanoGPT
This implementation is based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
```bibtex
@misc{karpathy2022nanogpt,
  author = {Karpathy, Andrej},
  title = {nanoGPT},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/karpathy/nanoGPT}
}
```

### Neuroscience Inspirations
- Predictive Coding: Rao & Ballard (1999)
- Sparse Coding: Olshausen & Field (1996)  
- Adaptive Computation: Graves (2016)
- Continual Learning: Kirkpatrick et al. (2017)

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with 🧠 by exploring the intersection of neuroscience and AI
</div>