"""
Hierarchical processing modules for multi-scale representation learning.
Inspired by the brain's hierarchical processing of information at different scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiScaleProcessor(nn.Module):
    """
    Processes information at multiple scales simultaneously.
    Lower scales capture local patterns, higher scales capture global context.
    """
    
    def __init__(self, n_embd, scales=[1, 2, 4, 8], combine_method='attention'):
        super().__init__()
        self.n_embd = n_embd
        self.scales = scales
        self.combine_method = combine_method
        
        # Create scale-specific processors
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_embd, n_embd, kernel_size=scale, stride=scale, padding=0),
                nn.GELU(),
                nn.Conv1d(n_embd, n_embd, kernel_size=1)
            ) for scale in scales
        ])
        
        # Upsampling layers to match original resolution
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose1d(n_embd, n_embd, kernel_size=scale, stride=scale)
            for scale in scales
        ])
        
        # Combination mechanism
        if combine_method == 'attention':
            self.combiner = nn.MultiheadAttention(n_embd, num_heads=8, batch_first=True)
        elif combine_method == 'weighted':
            self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        else:  # 'concat'
            self.combiner = nn.Linear(n_embd * len(scales), n_embd)
    
    def forward(self, x):
        """
        Process input at multiple scales.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
        
        Returns:
            Multi-scale representation [batch, seq_len, n_embd]
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Transpose for Conv1d
        x_conv = x.transpose(1, 2)  # [batch, n_embd, seq_len]
        
        scale_outputs = []
        for i, (scale, processor, upsampler) in enumerate(zip(self.scales, self.scale_processors, self.upsamplers)):
            # Process at current scale
            if scale == 1:
                scaled = processor(x_conv)
            else:
                # Pad if necessary
                pad_len = (scale - seq_len % scale) % scale
                if pad_len > 0:
                    x_padded = F.pad(x_conv, (0, pad_len))
                else:
                    x_padded = x_conv
                
                scaled = processor(x_padded)
                
                # Upsample back to original resolution
                scaled = upsampler(scaled)
                
                # Crop to original length
                scaled = scaled[:, :, :seq_len]
            
            scale_outputs.append(scaled.transpose(1, 2))  # Back to [batch, seq_len, n_embd]
        
        # Combine scales
        if self.combine_method == 'attention':
            # Stack along a new dimension for attention
            stacked = torch.stack(scale_outputs, dim=1)  # [batch, n_scales, seq_len, n_embd]
            stacked_flat = stacked.view(batch_size, -1, n_embd)  # [batch, n_scales * seq_len, n_embd]
            combined, _ = self.combiner(stacked_flat, stacked_flat, stacked_flat)
            combined = combined.view(batch_size, len(self.scales), seq_len, n_embd).mean(dim=1)
        elif self.combine_method == 'weighted':
            # Weighted combination
            weights = F.softmax(self.scale_weights, dim=0)
            combined = sum(w * out for w, out in zip(weights, scale_outputs))
        else:  # 'concat'
            # Concatenate and project
            concat = torch.cat(scale_outputs, dim=-1)
            combined = self.combiner(concat)
        
        return combined


class HierarchicalPooling(nn.Module):
    """
    Implements hierarchical pooling to create summary representations
    at different levels of abstraction.
    """
    
    def __init__(self, n_embd, pool_sizes=[2, 4, 8], pool_type='mean'):
        super().__init__()
        self.n_embd = n_embd
        self.pool_sizes = pool_sizes
        self.pool_type = pool_type
        
        # Projections for each pooling level
        self.pool_projections = nn.ModuleList([
            nn.Linear(n_embd, n_embd) for _ in pool_sizes
        ])
        
        # Gating mechanism for dynamic importance
        self.gates = nn.ModuleList([
            nn.Linear(n_embd, 1) for _ in pool_sizes
        ])
    
    def forward(self, x, return_all_levels=False):
        """
        Apply hierarchical pooling.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            return_all_levels: Whether to return all pooling levels
        
        Returns:
            Pooled representations
        """
        batch_size, seq_len, n_embd = x.shape
        
        pooled_outputs = []
        importance_scores = []
        
        for pool_size, projection, gate in zip(self.pool_sizes, self.pool_projections, self.gates):
            # Reshape for pooling
            pad_len = (pool_size - seq_len % pool_size) % pool_size
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_padded = x
            
            x_reshaped = x_padded.view(batch_size, -1, pool_size, n_embd)
            
            # Apply pooling
            if self.pool_type == 'mean':
                pooled = x_reshaped.mean(dim=2)
            elif self.pool_type == 'max':
                pooled = x_reshaped.max(dim=2)[0]
            else:  # 'attention'
                scores = gate(x_reshaped).softmax(dim=2)
                pooled = (x_reshaped * scores).sum(dim=2)
            
            # Project
            pooled = projection(pooled)
            pooled_outputs.append(pooled)
            
            # Calculate importance
            importance = torch.sigmoid(gate(pooled).mean(dim=1))
            importance_scores.append(importance)
        
        if return_all_levels:
            return pooled_outputs, importance_scores
        else:
            # Return weighted combination based on importance
            importance_weights = torch.stack(importance_scores, dim=1).softmax(dim=1)
            combined = sum(
                w.unsqueeze(1).unsqueeze(2) * out 
                for w, out in zip(importance_weights.unbind(1), pooled_outputs)
            )
            return combined


class TemporalHierarchy(nn.Module):
    """
    Implements temporal hierarchy for processing information at different
    time scales, similar to how the brain processes fast and slow dynamics.
    """
    
    def __init__(self, n_embd, n_layers=3, time_scales=[1, 4, 16]):
        super().__init__()
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.time_scales = time_scales
        
        # Create hierarchical layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_modules = nn.ModuleDict({
                'fast': nn.GRU(n_embd, n_embd, batch_first=True),
                'slow': nn.GRU(n_embd, n_embd, batch_first=True),
                'gate': nn.Linear(n_embd * 2, n_embd)
            })
            self.layers.append(layer_modules)
        
        # Cross-scale interaction
        self.cross_scale_attn = nn.MultiheadAttention(n_embd, num_heads=8, batch_first=True)
    
    def forward(self, x):
        """
        Process input through temporal hierarchy.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
        
        Returns:
            Hierarchically processed tensor
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Initialize states for different time scales
        fast_state = x
        slow_state = x
        
        for layer in self.layers:
            # Fast path (processes every timestep)
            fast_out, _ = layer['fast'](fast_state)
            
            # Slow path (processes downsampled sequence)
            # Downsample
            slow_input = slow_state[:, ::4, :]  # Take every 4th element
            slow_out, _ = layer['slow'](slow_input)
            
            # Upsample slow output
            slow_out_upsampled = F.interpolate(
                slow_out.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
            # Gate mechanism to combine fast and slow
            combined_input = torch.cat([fast_out, slow_out_upsampled], dim=-1)
            gate = torch.sigmoid(layer['gate'](combined_input))
            
            # Update states
            fast_state = gate * fast_out + (1 - gate) * slow_out_upsampled
            slow_state = fast_state
        
        # Cross-scale attention
        output, _ = self.cross_scale_attn(fast_state, slow_state, slow_state)
        
        return output


class PyramidalProcessing(nn.Module):
    """
    Implements pyramidal processing similar to cortical pyramidal neurons,
    integrating information across multiple dendritic branches.
    """
    
    def __init__(self, n_embd, n_branches=4, integration='nonlinear'):
        super().__init__()
        self.n_embd = n_embd
        self.n_branches = n_branches
        self.integration = integration
        
        # Dendritic branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.GELU(),
                nn.Linear(n_embd // 2, n_embd),
                nn.LayerNorm(n_embd)
            ) for _ in range(n_branches)
        ])
        
        # Integration mechanism
        if integration == 'nonlinear':
            self.integrator = nn.Sequential(
                nn.Linear(n_embd * n_branches, n_embd * 2),
                nn.GELU(),
                nn.Linear(n_embd * 2, n_embd)
            )
        else:  # 'linear'
            self.integrator = nn.Linear(n_embd * n_branches, n_embd)
        
        # Modulation gates for each branch
        self.branch_gates = nn.ModuleList([
            nn.Linear(n_embd, 1) for _ in range(n_branches)
        ])
    
    def forward(self, x):
        """
        Process through pyramidal structure.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
        
        Returns:
            Integrated output [batch, seq_len, n_embd]
        """
        # Process through each branch
        branch_outputs = []
        branch_weights = []
        
        for branch, gate in zip(self.branches, self.branch_gates):
            branch_out = branch(x)
            branch_outputs.append(branch_out)
            
            # Calculate branch importance
            weight = torch.sigmoid(gate(x))
            branch_weights.append(weight)
        
        # Normalize weights
        weights = torch.cat(branch_weights, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Apply weights to branches
        weighted_outputs = []
        for i, (branch_out, weight) in enumerate(zip(branch_outputs, weights.chunk(self.n_branches, dim=-1))):
            weighted_outputs.append(branch_out * weight)
        
        # Integrate branches
        if self.integration == 'nonlinear':
            concatenated = torch.cat(weighted_outputs, dim=-1)
            output = self.integrator(concatenated)
        else:
            output = sum(weighted_outputs)
        
        return output