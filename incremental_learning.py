"""
Incremental learning module for efficient knowledge updates.
Inspired by synaptic plasticity and memory consolidation in the brain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryBank:
    """Stores exemplars and their representations for memory replay."""
    examples: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    importance_scores: List[float] = None
    max_size: int = 1000
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []
            self.representations = []
            self.importance_scores = []
    
    def add(self, example: torch.Tensor, representation: torch.Tensor, importance: float):
        """Add a new example to memory bank."""
        self.examples.append(example.cpu())
        self.representations.append(representation.cpu())
        self.importance_scores.append(importance)
        
        # Maintain size limit
        if len(self.examples) > self.max_size:
            # Remove least important example
            min_idx = np.argmin(self.importance_scores)
            self.examples.pop(min_idx)
            self.representations.pop(min_idx)
            self.importance_scores.pop(min_idx)
    
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample examples weighted by importance."""
        if len(self.examples) == 0:
            return None, None
        
        # Sample indices based on importance
        probs = np.array(self.importance_scores)
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.examples), size=min(n_samples, len(self.examples)), 
                                 p=probs, replace=False)
        
        examples = torch.stack([self.examples[i] for i in indices])
        representations = torch.stack([self.representations[i] for i in indices])
        
        return examples, representations


class ElasticWeightConsolidation(nn.Module):
    """
    Implements Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    Inspired by synaptic consolidation in biological neural networks.
    """
    
    def __init__(self, model, lambda_ewc=0.1):
        super().__init__()
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Fisher information matrix
        self.fisher_dict = {}
        self.optimal_params_dict = {}
        
    def compute_fisher_information(self, dataloader, device):
        """Compute Fisher information matrix for current task."""
        self.fisher_dict = {}
        self.optimal_params_dict = {}
        
        self.model.eval()
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            self.model.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_dict[name] += param.grad.data ** 2
        
        # Normalize
        n_samples = len(dataloader)
        for name in self.fisher_dict:
            self.fisher_dict[name] /= n_samples
            self.optimal_params_dict[name] = self.model.state_dict()[name].clone()
    
    def ewc_loss(self):
        """Calculate EWC regularization loss."""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optimal_param = self.optimal_params_dict[name]
                loss += (fisher * (param - optimal_param) ** 2).sum()
        
        return self.lambda_ewc * loss


class ProgressiveNeuralNetwork(nn.Module):
    """
    Implements progressive neural networks that add new columns for new tasks
    while keeping previous columns frozen.
    """
    
    def __init__(self, base_model_config, n_tasks=5):
        super().__init__()
        self.base_config = base_model_config
        self.n_tasks = n_tasks
        self.current_task = 0
        
        # Create columns for each task
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        # Add first column
        self.add_new_column()
    
    def add_new_column(self):
        """Add a new column for a new task."""
        from model_efficient import EfficientBlock
        
        # Create new column
        column = nn.ModuleList([
            EfficientBlock(self.base_config, self.base_config.efficiency, i)
            for i in range(self.base_config.n_layer)
        ])
        self.columns.append(column)
        
        # Create lateral connections from previous columns
        if self.current_task > 0:
            lateral = nn.ModuleList()
            for layer_idx in range(self.base_config.n_layer):
                # Connections from all previous columns to current column
                layer_lateral = nn.ModuleList([
                    nn.Linear(self.base_config.n_embd, self.base_config.n_embd, bias=False)
                    for _ in range(self.current_task)
                ])
                lateral.append(layer_lateral)
            self.lateral_connections.append(lateral)
        
        self.current_task += 1
    
    def forward(self, x, task_id=None):
        """Forward pass through progressive network."""
        if task_id is None:
            task_id = self.current_task - 1
        
        # Store activations from all columns
        column_activations = [[] for _ in range(task_id + 1)]
        
        # Process through columns
        for col_idx in range(task_id + 1):
            h = x
            for layer_idx, block in enumerate(self.columns[col_idx]):
                # Add lateral connections from previous columns
                if col_idx > 0:
                    lateral_input = 0
                    for prev_col_idx in range(col_idx):
                        lateral_weight = self.lateral_connections[col_idx - 1][layer_idx][prev_col_idx]
                        lateral_input += lateral_weight(column_activations[prev_col_idx][layer_idx])
                    h = h + lateral_input
                
                h, _ = block(h)
                column_activations[col_idx].append(h)
        
        # Return final activation from target column
        return column_activations[task_id][-1]


class AdaptiveParameterGenerator(nn.Module):
    """
    Generates task-specific parameters dynamically.
    Inspired by neuromodulation in the brain.
    """
    
    def __init__(self, n_embd, n_tasks=10):
        super().__init__()
        self.n_embd = n_embd
        self.n_tasks = n_tasks
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(n_tasks, n_embd)
        
        # Parameter generators
        self.weight_generator = nn.Sequential(
            nn.Linear(n_embd, n_embd * 2),
            nn.GELU(),
            nn.Linear(n_embd * 2, n_embd * n_embd)
        )
        
        self.bias_generator = nn.Linear(n_embd, n_embd)
        
        # Gating mechanism
        self.gate_generator = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.Sigmoid()
        )
    
    def forward(self, x, task_id):
        """Generate task-specific parameters."""
        # Get task embedding
        task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
        
        # Generate parameters
        weight = self.weight_generator(task_emb).view(self.n_embd, self.n_embd)
        bias = self.bias_generator(task_emb)
        gate = self.gate_generator(task_emb)
        
        # Apply task-specific transformation
        x_transformed = F.linear(x, weight, bias)
        
        # Gate the transformation
        output = gate * x_transformed + (1 - gate) * x
        
        return output


class ContinualLearningWrapper(nn.Module):
    """
    Wrapper for models to add continual learning capabilities.
    """
    
    def __init__(self, base_model, method='ewc', memory_size=1000):
        super().__init__()
        self.base_model = base_model
        self.method = method
        self.memory_bank = MemoryBank(max_size=memory_size)
        
        if method == 'ewc':
            self.ewc = ElasticWeightConsolidation(base_model)
        elif method == 'progressive':
            # Would need to restructure base model
            raise NotImplementedError("Progressive networks require model restructuring")
        elif method == 'adaptive':
            self.param_generator = AdaptiveParameterGenerator(base_model.config.n_embd)
        
        self.current_task = 0
    
    def forward(self, idx, targets=None, task_id=None):
        """Forward pass with continual learning."""
        if task_id is None:
            task_id = self.current_task
        
        # Get base model output
        if self.method == 'adaptive' and task_id is not None:
            # Apply task-specific modulation
            logits, loss = self.base_model(idx, targets)
            
            # Modulate hidden states (would need to modify base model for full integration)
            # This is a simplified version
            if hasattr(self.base_model, 'transformer'):
                h = self.base_model.transformer.ln_f.weight
                h_modulated = self.param_generator(h.unsqueeze(0), task_id).squeeze(0)
                self.base_model.transformer.ln_f.weight.data = h_modulated
        else:
            logits, loss = self.base_model(idx, targets)
        
        # Add EWC regularization if training
        if self.method == 'ewc' and loss is not None and self.training:
            ewc_loss = self.ewc.ewc_loss()
            loss = loss + ewc_loss
        
        # Store important examples
        if self.training and loss is not None:
            importance = loss.item()  # Simple importance metric
            self.memory_bank.add(idx[0], logits[0].detach(), importance)
        
        return logits, loss
    
    def consolidate_task(self, dataloader, device):
        """Consolidate knowledge after training on a task."""
        if self.method == 'ewc':
            self.ewc.compute_fisher_information(dataloader, device)
        
        self.current_task += 1
    
    def rehearsal_batch(self, n_samples=32):
        """Get a batch of examples for rehearsal."""
        return self.memory_bank.sample(n_samples)


class OnlineLearningOptimizer(nn.Module):
    """
    Implements online learning with adaptive learning rates.
    Inspired by synaptic plasticity rules.
    """
    
    def __init__(self, base_optimizer, meta_lr=0.01):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.meta_lr = meta_lr
        
        # Track parameter-specific learning rates
        self.param_lrs = {}
        for group in base_optimizer.param_groups:
            for p in group['params']:
                self.param_lrs[id(p)] = nn.Parameter(torch.tensor(group['lr']))
    
    def step(self, loss):
        """Perform optimization step with adaptive learning rates."""
        # Standard optimization step
        self.base_optimizer.step()
        
        # Meta-learning step for learning rates
        # (Simplified version - full implementation would use gradients)
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Adjust learning rate based on gradient magnitude
                    grad_norm = p.grad.norm().item()
                    lr_param = self.param_lrs[id(p)]
                    
                    # Simple adaptive rule
                    if grad_norm > 1.0:
                        lr_param.data *= 0.99
                    else:
                        lr_param.data *= 1.01
                    
                    # Clip learning rate
                    lr_param.data.clamp_(1e-5, 1e-2)
    
    def get_param_lr(self, param):
        """Get current learning rate for a parameter."""
        return self.param_lrs[id(param)].item()