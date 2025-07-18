"""
Brain-inspired efficiency modules for nanoGPT.

This module implements various brain-inspired mechanisms to improve computational efficiency:
1. Predictive processing with early exit
2. Sparse activation patterns
3. Adaptive computation time
4. Dynamic routing
5. Memory-efficient attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency features."""
    # Predictive processing
    enable_early_exit: bool = False
    exit_threshold: float = 0.95  # Confidence threshold for early exit
    min_exit_layer: int = 4  # Minimum layer before allowing exit
    
    # Sparse activation
    enable_sparse_activation: bool = False
    sparsity_ratio: float = 0.1  # Top 10% of neurons active
    
    # Adaptive computation time
    enable_adaptive_compute: bool = False
    max_pondering_steps: int = 5
    pondering_threshold: float = 0.01
    
    # Dynamic routing
    enable_dynamic_routing: bool = False
    routing_temperature: float = 1.0
    
    # Memory-efficient attention
    enable_local_attention: bool = False
    local_attention_window: int = 256
    
    # Hierarchical processing
    enable_hierarchical: bool = False
    hierarchical_type: str = 'multi_scale'  # 'multi_scale', 'temporal', 'pyramidal'
    hierarchical_scales: List[int] = None  # Default: [1, 2, 4, 8]
    
    # Monitoring
    track_efficiency_metrics: bool = True
    
    def __post_init__(self):
        if self.hierarchical_scales is None:
            self.hierarchical_scales = [1, 2, 4, 8]


class PredictiveProcessor(nn.Module):
    """
    Implements predictive processing with early exit capability.
    Inspired by the brain's ability to make quick predictions and only engage
    deeper processing when necessary.
    """
    
    def __init__(self, n_embd, n_head, bias=False):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        
        # Prediction head for each layer
        self.predictor = nn.Linear(n_embd, n_embd, bias=bias)
        self.confidence_head = nn.Linear(n_embd, 1, bias=bias)
        
    def forward(self, x, return_confidence=False):
        """
        Predicts the next representation and confidence level.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            return_confidence: Whether to return confidence scores
            
        Returns:
            prediction: Predicted representation
            confidence: Confidence scores (if requested)
        """
        prediction = self.predictor(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        if return_confidence:
            return prediction, confidence
        return prediction


class SparseActivation(nn.Module):
    """
    Implements sparse activation patterns similar to the brain's sparse coding.
    Only activates top-k neurons based on their activation strength.
    """
    
    def __init__(self, sparsity_ratio=0.1):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        
    def forward(self, x):
        """
        Apply sparse activation by keeping only top-k activations.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            
        Returns:
            Sparsely activated tensor
        """
        batch_size, seq_len, n_embd = x.shape
        k = max(1, int(n_embd * self.sparsity_ratio))
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x.abs(), k, dim=-1)
        
        # Create sparse mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Apply mask while preserving gradients
        sparse_x = x * mask
        
        # Scale to maintain expected activation magnitude
        scale = n_embd / k
        return sparse_x * scale


class AdaptiveComputationTime(nn.Module):
    """
    Implements adaptive computation time allowing the model to "ponder"
    difficult inputs longer, similar to how humans spend more time on
    complex problems.
    """
    
    def __init__(self, n_embd, max_steps=5, threshold=0.01):
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Pondering network
        self.ponder_gru = nn.GRUCell(n_embd, n_embd)
        self.halting_head = nn.Linear(n_embd, 1)
        
    def forward(self, x):
        """
        Apply adaptive computation time to input.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            
        Returns:
            output: Processed tensor
            ponder_steps: Number of pondering steps taken
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Reshape for GRU processing
        x_flat = x.view(-1, n_embd)
        
        output = torch.zeros_like(x_flat)
        halting_prob = torch.zeros(x_flat.shape[0], 1, device=x.device)
        remainders = torch.ones(x_flat.shape[0], 1, device=x.device)
        n_updates = torch.zeros(x_flat.shape[0], 1, device=x.device)
        
        state = x_flat
        
        for step in range(self.max_steps):
            # Update state
            state = self.ponder_gru(x_flat, state)
            
            # Compute halting probability
            p = torch.sigmoid(self.halting_head(state))
            
            # Compute mask for elements still processing
            still_running = (halting_prob < 1.0 - self.threshold).float()
            
            # Update halting probability and output
            new_halted = (halting_prob + p * still_running > self.threshold).float()
            halting_prob = halting_prob + p * still_running
            
            # Update output with weighted contribution
            output = output + state * p * still_running
            n_updates = n_updates + still_running
            
            # Check if all sequences have halted
            if (still_running.sum() == 0).item():
                break
        
        # Add remainder
        output = output + state * remainders
        
        # Reshape back
        output = output.view(batch_size, seq_len, n_embd)
        
        return output, n_updates.mean()


class DynamicRouter(nn.Module):
    """
    Implements dynamic routing to conditionally execute computation paths,
    similar to how the brain routes information through different pathways
    based on content.
    """
    
    def __init__(self, n_embd, n_experts=4, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.temperature = temperature
        
        # Router network
        self.router = nn.Linear(n_embd, n_experts)
        
        # Expert networks (simplified as linear transformations)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, n_embd * 2),
                nn.GELU(),
                nn.Linear(n_embd * 2, n_embd)
            ) for _ in range(n_experts)
        ])
        
    def forward(self, x):
        """
        Route input through expert networks based on content.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            
        Returns:
            output: Processed tensor
            routing_weights: Routing weights for analysis
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Compute routing weights
        routing_logits = self.router(x) / self.temperature
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Process through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, n_embd, n_experts]
        
        # Weighted combination
        routing_weights_expanded = routing_weights.unsqueeze(-2)  # [batch, seq, 1, n_experts]
        output = (expert_outputs * routing_weights_expanded).sum(dim=-1)
        
        return output, routing_weights


class LocalAttention(nn.Module):
    """
    Implements local attention patterns to reduce computational complexity
    from O(n²) to O(n*w) where w is the window size.
    Inspired by the brain's local connectivity patterns.
    """
    
    def __init__(self, window_size=256):
        super().__init__()
        self.window_size = window_size
        
    def create_local_attention_mask(self, seq_len, device):
        """Create a mask for local attention."""
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0
            
        # Ensure causal masking
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        mask = mask + causal_mask
        
        return mask
    
    def apply_local_attention(self, scores, seq_len):
        """Apply local attention masking to attention scores."""
        local_mask = self.create_local_attention_mask(seq_len, scores.device)
        scores = scores + local_mask.unsqueeze(0).unsqueeze(0)
        return scores


class EfficiencyMetrics:
    """Tracks efficiency metrics during forward passes."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.early_exit_layers = []
        self.sparsity_levels = []
        self.ponder_steps = []
        self.routing_entropy = []
        self.total_flops_saved = 0
        
    def log_early_exit(self, layer):
        self.early_exit_layers.append(layer)
        
    def log_sparsity(self, sparsity):
        self.sparsity_levels.append(sparsity)
        
    def log_ponder_steps(self, steps):
        self.ponder_steps.append(steps)
        
    def log_routing_entropy(self, weights):
        # Calculate entropy of routing weights
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        self.routing_entropy.append(entropy.item())
        
    def estimate_flops_saved(self, config):
        """Estimate FLOPs saved by efficiency features."""
        saved = 0
        
        # Early exit saves
        if self.early_exit_layers:
            avg_exit = sum(self.early_exit_layers) / len(self.early_exit_layers)
            saved += (1 - avg_exit / config.n_layer) * 0.5  # Approximate 50% saving
            
        # Sparsity saves
        if self.sparsity_levels:
            avg_sparsity = sum(self.sparsity_levels) / len(self.sparsity_levels)
            saved += (1 - avg_sparsity) * 0.3  # Approximate 30% saving in MLP
            
        self.total_flops_saved = saved
        return saved
    
    def get_summary(self):
        """Get summary of efficiency metrics."""
        summary = {
            'avg_exit_layer': sum(self.early_exit_layers) / len(self.early_exit_layers) if self.early_exit_layers else None,
            'avg_sparsity': sum(self.sparsity_levels) / len(self.sparsity_levels) if self.sparsity_levels else None,
            'avg_ponder_steps': sum(self.ponder_steps) / len(self.ponder_steps) if self.ponder_steps else None,
            'avg_routing_entropy': sum(self.routing_entropy) / len(self.routing_entropy) if self.routing_entropy else None,
            'estimated_flops_saved': self.total_flops_saved
        }
        return summary