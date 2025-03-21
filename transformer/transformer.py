import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_sequence_length):
        super(SinusoidalPositionalEncoding, self).__init__()

        assert d_embedding % 2 == 0
        self.max_sequence_length = max_sequence_length
        position_encodings = torch.zeros((max_sequence_length, d_embedding))
        numerator = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(-1).repeat(1, d_embedding // 2)
        denominator = torch.arange(0, d_embedding, 2, dtype=torch.float).unsqueeze(0).repeat(max_sequence_length, 1)
        denominator = torch.pow(1e4, denominator / d_embedding)

        position_encodings[:, 0::2] = torch.sin(numerator / denominator)
        position_encodings[:, 1::2] = torch.cos(numerator / denominator)

        self.register_buffer('position_encodings', position_encodings.unsqueeze(0))

    def forward(self, embeddings):
        _, num_rows, _ = embeddings.shape
        assert num_rows <= self.max_sequence_length
        return embeddings + self.position_encodings[:, :num_rows]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embedding, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Need to make sure that number of heads divides the embedding dimension

        assert d_embedding % num_heads == 0
        self.d_embedding = d_embedding
        self.d_head = d_embedding // num_heads
        self.num_heads = num_heads

        self.Q = nn.Linear(d_embedding, d_embedding)
        self.K = nn.Linear(d_embedding, d_embedding)
        self.V = nn.Linear(d_embedding, d_embedding)
        self.O = nn.Linear(d_embedding, d_embedding)

    def split_heads(self, X):
        d_batch, d_trajectory, d_embedding = X.shape
        assert d_embedding == self.d_embedding
        return X.reshape(d_batch, d_trajectory, self.num_heads, self.d_head).transpose(1, 2)

    def combine_heads(self, X):
        d_batch, num_heads, d_trajectory, d_head = X.shape
        assert num_heads == self.num_heads and d_head == self.d_head
        return X.transpose(1, 2).reshape(d_batch, d_trajectory, self.d_embedding)

    def get_attention_probabilities(self, queries, keys, mask):
        logits = queries @ keys.transpose(-1, -2) / math.sqrt(self.d_head)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        attention_probs = logits.softmax(dim=-1)
        return attention_probs

    def forward(self, queries, values, mask = None):
        # For each query, create a new embedding by looking at values
        queries = self.split_heads(self.Q(queries))
        keys = self.split_heads(self.K(values))
        values = self.split_heads(self.V(values))

        # Get the new values
        values = self.get_attention_probabilities(queries, keys, mask) @ values
        return self.O(self.combine_heads(values))


class FeedforwardNetwork(nn.Module):
    def __init__(self, d_embedding, d_hidden):
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_embedding, d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, d_embedding)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_embedding, num_heads, d_hidden, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_embedding, num_heads)
        self.feed_forward = FeedforwardNetwork(d_embedding, d_hidden)
        self.norm1 = nn.LayerNorm(d_embedding)
        self.norm2 = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embeddings, mask):
        self_attention_emb = self.self_attention(embeddings, embeddings, mask)
        embeddings = self.norm1(embeddings + self.dropout(self_attention_emb))
        feedforward_emb = self.feed_forward(embeddings)
        return self.norm2(embeddings + self.dropout(feedforward_emb))


class CausalDecoderTransformer(nn.Module):
    def __init__(self, cfg):
        super(CausalDecoderTransformer, self).__init__()
        d_embedding = cfg.model.d_embedding
        self.max_new_tokens = cfg.validation.max_new_tokens
        self.chunk_size = cfg.model.chunk_size
        self.embedding = nn.Embedding(cfg.tokenization.vocab_size, d_embedding)
        self.transformer_layers = \
            nn.ModuleList([TransformerLayer(d_embedding, cfg.model.num_heads, cfg.model.d_hidden, cfg.model.dropout_rate)
                for _ in range(cfg.model.num_layers)])
        self.positional_encoder = SinusoidalPositionalEncoding(d_embedding, self.chunk_size)
        self.output_logit_layer = nn.Linear(d_embedding, cfg.tokenization.vocab_size)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)

    def generate_causal_mask(self, size):
        return torch.tril(torch.ones(size, size)).to(device)

    def forward(self, sequence):
        embeddings = self.dropout(self.positional_encoder(self.embedding(sequence)))
        mask = self.generate_causal_mask(embeddings.shape[1])
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, mask)
        output_logits = self.output_logit_layer(embeddings)
        return output_logits

    @torch.no_grad()
    def generate_text(self, prompt, is_eos, temperature):
            for _ in range(self.max_new_tokens):
                context = (prompt if prompt.size(1) < self.chunk_size else prompt[:, -self.chunk_size:])
                logits = self(context)
                if temperature == 0:
                    logits = logits[:, -1, :]
                    next_token = torch.argmax(logits).reshape(1, 1)
                else:
                    logits = logits[:, -1, :] / temperature
                    probs = nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).reshape(1, 1)

                if is_eos(next_token.item()):
                    break
                prompt = torch.cat([prompt, next_token], dim=-1)
            return prompt















