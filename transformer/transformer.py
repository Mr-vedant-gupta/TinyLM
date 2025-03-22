from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import DictConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalPositionalEncoding(nn.Module):
    """Implements sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_embedding: int, max_sequence_length: int):
        """Initialize positional encoding.
        
        Args:
            d_embedding: Embedding dimension (must be even)
            max_sequence_length: Maximum sequence length to encode
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        if d_embedding % 2 != 0:
            raise ValueError("Embedding dimension must be even")
            
        self.max_sequence_length = max_sequence_length
        position_encodings = torch.zeros((max_sequence_length, d_embedding))
        
        # Compute position encodings using sin/cos
        numerator = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(-1).repeat(1, d_embedding // 2)
        denominator = torch.arange(0, d_embedding, 2, dtype=torch.float).unsqueeze(0).repeat(max_sequence_length, 1)
        denominator = torch.pow(1e4, denominator / d_embedding)

        position_encodings[:, 0::2] = torch.sin(numerator / denominator)
        position_encodings[:, 1::2] = torch.cos(numerator / denominator)

        self.register_buffer('position_encodings', position_encodings.unsqueeze(0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input embeddings."""
        _, num_rows, _ = embeddings.shape
        if num_rows > self.max_sequence_length:
            raise ValueError(f"Sequence length {num_rows} exceeds maximum {self.max_sequence_length}")
        return embeddings + self.position_encodings[:, :num_rows]


class MultiHeadAttention(nn.Module):
    """Implements multi-head self-attention mechanism."""
    
    def __init__(self, d_embedding: int, num_heads: int):
        """Initialize multi-head attention.
        
        Args:
            d_embedding: Embedding dimension
            num_heads: Number of attention heads (must divide d_embedding)
        """
        super(MultiHeadAttention, self).__init__()
        if d_embedding % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
            
        self.d_embedding = d_embedding
        self.d_head = d_embedding // num_heads
        self.num_heads = num_heads

        self.Q = nn.Linear(d_embedding, d_embedding)
        self.K = nn.Linear(d_embedding, d_embedding)
        self.V = nn.Linear(d_embedding, d_embedding)
        self.O = nn.Linear(d_embedding, d_embedding)

    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention computation."""
        d_batch, d_trajectory, d_embedding = X.shape
        if d_embedding != self.d_embedding:
            raise ValueError(f"Expected embedding dim {self.d_embedding}, got {d_embedding}")
        return X.reshape(d_batch, d_trajectory, self.num_heads, self.d_head).transpose(1, 2)

    def combine_heads(self, X: torch.Tensor) -> torch.Tensor:
        """Combine multi-head attention outputs."""
        d_batch, num_heads, d_trajectory, d_head = X.shape
        if num_heads != self.num_heads or d_head != self.d_head:
            raise ValueError("Invalid head dimensions")
        return X.transpose(1, 2).reshape(d_batch, d_trajectory, self.d_embedding)

    def get_attention_probabilities(self, queries: torch.Tensor, keys: torch.Tensor, 
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention probabilities with optional masking."""
        logits = queries @ keys.transpose(-1, -2) / math.sqrt(self.d_head)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits.softmax(dim=-1)

    def forward(self, queries: torch.Tensor, values: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head attention."""
        queries = self.split_heads(self.Q(queries))
        keys = self.split_heads(self.K(values))
        values = self.split_heads(self.V(values))

        values = self.get_attention_probabilities(queries, keys, mask) @ values
        return self.O(self.combine_heads(values))


class FeedforwardNetwork(nn.Module):
    """Implements the feed-forward network used in transformer layers."""
    
    def __init__(self, d_embedding: int, d_hidden: int):
        """Initialize feed-forward network.
        
        Args:
            d_embedding: Input/output dimension
            d_hidden: Hidden layer dimension
        """
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_embedding, d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, d_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.linear2(self.relu(self.linear1(x)))


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network."""
    
    def __init__(self, d_embedding: int, num_heads: int, d_hidden: int, dropout_rate: float):
        """Initialize transformer layer.
        
        Args:
            d_embedding: Embedding dimension
            num_heads: Number of attention heads
            d_hidden: Feed-forward hidden dimension
            dropout_rate: Dropout probability
        """
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_embedding, num_heads)
        self.feed_forward = FeedforwardNetwork(d_embedding, d_hidden)
        self.norm1 = nn.LayerNorm(d_embedding)
        self.norm2 = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer layer."""
        self_attention_emb = self.self_attention(embeddings, embeddings, mask)
        embeddings = self.norm1(embeddings + self.dropout(self_attention_emb))
        feedforward_emb = self.feed_forward(embeddings)
        return self.norm2(embeddings + self.dropout(feedforward_emb))


class CausalDecoderTransformer(nn.Module):
    """Causal decoder-only transformer model for language modeling."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the transformer model.
        
        Args:
            cfg: Configuration object containing model parameters
        """
        super(CausalDecoderTransformer, self).__init__()
        d_embedding = cfg.model.d_embedding
        self.max_new_tokens = cfg.validation.max_new_tokens
        self.chunk_size = cfg.model.chunk_size
        
        self.embedding = nn.Embedding(cfg.tokenization.vocab_size, d_embedding)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_embedding, cfg.model.num_heads, cfg.model.d_hidden, cfg.model.dropout_rate)
            for _ in range(cfg.model.num_layers)
        ])
        self.positional_encoder = SinusoidalPositionalEncoding(d_embedding, self.chunk_size)
        self.output_logit_layer = nn.Linear(d_embedding, cfg.tokenization.vocab_size)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)

    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        return torch.tril(torch.ones(size, size)).to(device)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        embeddings = self.dropout(self.positional_encoder(self.embedding(sequence)))
        mask = self.generate_causal_mask(embeddings.shape[1])
        
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, mask)
            
        return self.output_logit_layer(embeddings)

    @torch.no_grad()
    def generate_text(self, prompt: torch.Tensor, is_eos: callable, temperature: float) -> torch.Tensor:
        """Generate text from a prompt using temperature sampling.
        
        Args:
            prompt: Input token sequence
            is_eos: Function to check for end-of-sequence token
            temperature: Sampling temperature (0 for greedy decoding)
            
        Returns:
            Generated token sequence
        """
        for _ in range(self.max_new_tokens):
            context = (prompt if prompt.size(1) < self.chunk_size
                    else prompt[:, -self.chunk_size:])
            logits = self(context)

            if temperature == 0:
                next_token = torch.argmax(logits[:, -1, :]).reshape(1, 1)
            else:
                logits = logits[:, -1, :] / temperature
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).reshape(1, 1)

            if is_eos(next_token.item()):
                break
            prompt = torch.cat([prompt, next_token], dim=-1)
            
        return prompt















