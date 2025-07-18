"""
Efficient GPT Language Model with brain-inspired optimizations.
This extends the base nanoGPT model with various efficiency improvements.
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import LayerNorm, GPTConfig as BaseGPTConfig
from efficient_modules import (
    EfficiencyConfig, PredictiveProcessor, SparseActivation,
    AdaptiveComputationTime, DynamicRouter, LocalAttention,
    EfficiencyMetrics
)
from hierarchical_modules import (
    MultiScaleProcessor, HierarchicalPooling, TemporalHierarchy, PyramidalProcessing
)


class EfficientCausalSelfAttention(nn.Module):
    """Enhanced self-attention with optional local attention patterns."""
    
    def __init__(self, config, efficiency_config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Standard attention components
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Efficiency components
        self.efficiency_config = efficiency_config
        if efficiency_config.enable_local_attention:
            self.local_attention = LocalAttention(efficiency_config.local_attention_window)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply attention
        if self.flash and not self.efficiency_config.enable_local_attention:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            if self.efficiency_config.enable_local_attention:
                att = self.local_attention.apply_local_attention(att, T)
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class EfficientMLP(nn.Module):
    """MLP with optional sparse activation and dynamic routing."""
    
    def __init__(self, config, efficiency_config):
        super().__init__()
        self.efficiency_config = efficiency_config
        
        if efficiency_config.enable_dynamic_routing:
            self.router = DynamicRouter(config.n_embd, n_experts=4, 
                                      temperature=efficiency_config.routing_temperature)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
            
        if efficiency_config.enable_sparse_activation:
            self.sparse_activation = SparseActivation(efficiency_config.sparsity_ratio)
    
    def forward(self, x, metrics=None):
        if self.efficiency_config.enable_dynamic_routing:
            x, routing_weights = self.router(x)
            if metrics:
                metrics.log_routing_entropy(routing_weights)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            
            if self.efficiency_config.enable_sparse_activation:
                x = self.sparse_activation(x)
                if metrics:
                    sparsity = (x == 0).float().mean().item()
                    metrics.log_sparsity(sparsity)
            
            x = self.c_proj(x)
            x = self.dropout(x)
        
        return x


class EfficientBlock(nn.Module):
    """Transformer block with efficiency enhancements."""
    
    def __init__(self, config, efficiency_config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.efficiency_config = efficiency_config
        
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = EfficientCausalSelfAttention(config, efficiency_config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = EfficientMLP(config, efficiency_config)
        
        # Predictive processing for early exit
        if efficiency_config.enable_early_exit:
            self.predictor = PredictiveProcessor(config.n_embd, config.n_head, config.bias)
        
        # Adaptive computation time
        if efficiency_config.enable_adaptive_compute:
            self.act = AdaptiveComputationTime(config.n_embd, 
                                              efficiency_config.max_pondering_steps,
                                              efficiency_config.pondering_threshold)
        
        # Hierarchical processing
        if efficiency_config.enable_hierarchical:
            if efficiency_config.hierarchical_type == 'multi_scale':
                self.hierarchical = MultiScaleProcessor(config.n_embd, 
                                                       scales=efficiency_config.hierarchical_scales)
            elif efficiency_config.hierarchical_type == 'temporal':
                self.hierarchical = TemporalHierarchy(config.n_embd)
            elif efficiency_config.hierarchical_type == 'pyramidal':
                self.hierarchical = PyramidalProcessing(config.n_embd)
    
    def forward(self, x, metrics=None):
        # Apply predictive processing if enabled
        if self.efficiency_config.enable_early_exit and self.layer_idx >= self.efficiency_config.min_exit_layer:
            prediction, confidence = self.predictor(x, return_confidence=True)
            
            # Check if we can exit early
            max_confidence = confidence.max().item()
            if max_confidence > self.efficiency_config.exit_threshold:
                if metrics:
                    metrics.log_early_exit(self.layer_idx)
                return x, True  # Signal early exit
        
        # Standard transformer block computation
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), metrics)
        
        # Apply adaptive computation time if enabled
        if self.efficiency_config.enable_adaptive_compute:
            x, ponder_steps = self.act(x)
            if metrics:
                metrics.log_ponder_steps(ponder_steps.item())
        
        # Apply hierarchical processing if enabled
        if self.efficiency_config.enable_hierarchical:
            x = self.hierarchical(x)
        
        return x, False  # No early exit


@dataclass
class EfficientGPTConfig(BaseGPTConfig):
    """Configuration for Efficient GPT model."""
    # All base config fields are inherited
    # Add efficiency config
    efficiency: Optional[EfficiencyConfig] = None


class EfficientGPT(nn.Module):
    """GPT model with brain-inspired efficiency improvements."""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Initialize efficiency config if not provided
        if config.efficiency is None:
            config.efficiency = EfficiencyConfig()
        self.efficiency_config = config.efficiency
        
        # Create efficiency metrics tracker
        self.metrics = EfficiencyMetrics() if config.efficiency.track_efficiency_metrics else None
        
        # Build transformer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                EfficientBlock(config, config.efficiency, i) 
                for i in range(config.n_layer)
            ]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Reset metrics for this forward pass
        if self.metrics:
            self.metrics.reset()
        
        # Embed tokens and positions
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Process through transformer blocks
        early_exit = False
        for i, block in enumerate(self.transformer.h):
            x, early_exit = block(x, self.metrics)
            if early_exit:
                # Skip remaining layers
                break
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Calculate loss if targets provided
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def get_efficiency_summary(self):
        """Get summary of efficiency metrics from last forward pass."""
        if self.metrics:
            return self.metrics.get_summary()
        return None
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization including efficiency savings."""
        # Base MFU calculation
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Adjust for efficiency savings
        if self.metrics:
            flops_saved = self.metrics.estimate_flops_saved(cfg)
            flops_per_iter *= (1 - flops_saved)
        
        # Calculate MFU
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens with efficiency features active."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW optimizer with weight decay."""
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained model and add efficiency features."""
        # This would need to be implemented to convert standard GPT to efficient version
        raise NotImplementedError("Pretrained loading for EfficientGPT not yet implemented")