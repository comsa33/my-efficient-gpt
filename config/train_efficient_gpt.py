# Configuration for training Efficient GPT with brain-inspired optimizations

# Base model configuration
out_dir = 'out-efficient-shakespeare'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'efficient-gpt'
wandb_run_name = 'efficient-shakespeare'

# Data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Efficiency features configuration
use_efficient_model = True
efficiency_config = {
    # Predictive processing with early exit
    'enable_early_exit': True,
    'exit_threshold': 0.95,  # Confidence threshold for early exit
    'min_exit_layer': 2,     # Don't exit before this layer
    
    # Sparse activation
    'enable_sparse_activation': True,
    'sparsity_ratio': 0.1,   # Keep top 10% of activations
    
    # Adaptive computation time
    'enable_adaptive_compute': True,
    'max_pondering_steps': 3,
    'pondering_threshold': 0.01,
    
    # Dynamic routing
    'enable_dynamic_routing': True,
    'routing_temperature': 1.0,
    
    # Local attention
    'enable_local_attention': True,
    'local_attention_window': 128,
    
    # Metrics tracking
    'track_efficiency_metrics': True
}

# Training modes for experimentation
# Options: 'full', 'early_exit_only', 'sparse_only', 'local_attention_only', 'baseline'
efficiency_mode = 'full'

if efficiency_mode == 'early_exit_only':
    efficiency_config = {
        'enable_early_exit': True,
        'exit_threshold': 0.95,
        'min_exit_layer': 2,
        'track_efficiency_metrics': True
    }
elif efficiency_mode == 'sparse_only':
    efficiency_config = {
        'enable_sparse_activation': True,
        'sparsity_ratio': 0.1,
        'track_efficiency_metrics': True
    }
elif efficiency_mode == 'local_attention_only':
    efficiency_config = {
        'enable_local_attention': True,
        'local_attention_window': 128,
        'track_efficiency_metrics': True
    }
elif efficiency_mode == 'baseline':
    use_efficient_model = False
    efficiency_config = None