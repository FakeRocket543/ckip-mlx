"""Minimal BERT encoder in MLX for CKIP token classification."""

import mlx.core as mx
import mlx.nn as nn


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_pos, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, input_ids, token_type_ids=None):
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        x = self.word_embeddings(input_ids) + \
            self.position_embeddings(position_ids) + \
            self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(x)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        h = self.num_attention_heads
        d = self.attention_head_size
        q = self.query(x).reshape(B, L, h, d).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, h, d).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, h, d).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / (d ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return out


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        # self-attention
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_output_dense = nn.Linear(hidden_size, hidden_size)
        self.attention_output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # FFN
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, x, mask=None):
        attn = self.attention(x, mask)
        attn = self.attention_output_dense(attn)
        x = self.attention_output_LayerNorm(attn + x)
        h = nn.gelu(self.intermediate_dense(x))
        h = self.output_dense(h)
        return self.output_LayerNorm(h + x)


class BertForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs = config["hidden_size"]
        self.embeddings = BertEmbeddings(
            config["vocab_size"], hs,
            config["max_position_embeddings"])
        self.layers = [
            BertLayer(hs, config["num_attention_heads"],
                      config["intermediate_size"])
            for _ in range(config["num_hidden_layers"])
        ]
        self.classifier = nn.Linear(hs, config["num_labels"])

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.embeddings(input_ids, token_type_ids)
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
        for layer in self.layers:
            x = layer(x, mask)
        return self.classifier(x)
