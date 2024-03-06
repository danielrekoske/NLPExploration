import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

class NewGELU(nn.Module):
    def forward(self, x):
        """
        Applies the Gaussian Error Linear Unit (GELU) activation function to the input tensor x.
        
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying GELU activation.
        """
        # Implementation of the GELU activation function
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        """
        Implements a causal self-attention mechanism based on the given configuration.
        
        Args:
            config: Configuration object containing parameters for the attention mechanism.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Linear transformation for the attention mechanism
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Generating a causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        Forward pass through the causal self-attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C is the number of channels.
        
        Returns:
            torch.Tensor: Output tensor after applying the causal self-attention mechanism.
        """
        B, T, C = x.size()

        # Linear transformation to obtain queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshaping and transposing to prepare for matrix multiplication
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention calculation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Masking future information
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax normalization
        att = F.softmax(att, dim=-1)
        
        # Weighted sum of values
        y = att @ v
        
        # Reshaping and transposing back to original shape
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear transformation
        y = self.c_proj(y)
        
        return y

class Block(nn.Module):
    """An unassuming Transformer block."""

    def __init__(self, config):
        """
        Initializes a Transformer block.
        
        Args:
            config: Configuration object containing parameters for the block.
        """
        super().__init__()
        
        # Layer normalization for the attention mechanism
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # Causal self-attention mechanism
        self.attn = CausalSelfAttention(config)
        
        # Layer normalization for the feedforward network
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # Multi-layer perceptron (MLP) for the feedforward network
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        
        # Function to apply the feedforward network
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x):
        """
        Forward pass through the Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C is the number of channels.
        
        Returns:
            torch.Tensor: Output tensor after passing through the Transformer block.
        """
        # Apply attention mechanism followed by layer normalization
        x = x + self.attn(self.ln_1(x))
        
        # Apply feedforward network followed by layer normalization
        x = x + self.mlpf(self.ln_2(x))
        
        return x

class Transformer(nn.Module):
    """Transformer Language Model, exactly as seen in GPT-2"""

    def __init__(self, config):
        """
        Initializes a Transformer Language Model.
        
        Args:
            config: Configuration object containing parameters for the model.
        """
        super().__init__()
        self.block_size = config.block_size

        # Components of the Transformer model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Linear transformation for output prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Counting parameters in the model
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        """
        Returns the block size of the model.
        
        Returns:
            int: Block size.
        """
        return self.block_size

    def forward(self, idx, targets=None):
        """
        Forward pass through the Transformer Language Model.
        
        Args:
            idx (torch.Tensor): Input tensor of token indices with shape (B, T), where B is the batch size and T is the sequence length.
            targets (torch.Tensor): Target tensor of token indices with shape (B, T).
        
        Returns:
            tuple: Tuple containing logits tensor and optional loss tensor.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token embedding
        tok_emb = self.transformer.wte(idx)
        
        # Positional embedding
        pos_emb = self.transformer.wpe(pos)
        
        # Add token and positional embeddings
        x = tok_emb + pos_emb
        
        # Forward pass through each Transformer block
        for block in self.transformer.h:
            x = block(x)
        
        # Layer normalization
        x = self.transformer.ln_f(x)
        
        # Prediction
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
