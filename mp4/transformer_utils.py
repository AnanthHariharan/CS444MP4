import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, input_dim, query_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()
        assert(query_dim == key_dim)
        self.query_dim = query_dim
        self.input_dim = input_dim

        self.W_query = nn.Linear(input_dim, query_dim)
        self.W_key = nn.Linear(input_dim, key_dim)
        self.W_value = nn.Linear(input_dim, value_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        scale = torch.sqrt(torch.tensor(self.query_dim, dtype=queries.dtype, device=queries.device))
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale

        attention_weights = self.softmax(attention_scores)
        attn_output = torch.bmm(attention_weights, values)

        return attn_output


class LayerNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.w = nn.Parameter(torch.ones(self.input_dim))
        self.b = nn.Parameter(torch.zeros(self.input_dim))
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=True)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        x_out = x_normalized * self.w + self.b

        return x_out
