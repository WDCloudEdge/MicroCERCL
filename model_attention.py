import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        # Compute attention scores
        query_proj = self.query(queries)
        key_proj = self.key(keys)
        value_proj = self.value(values)

        # Compute attention weights
        attention_scores = torch.matmul(query_proj, key_proj.transpose(0, 1))
        attention_weights = self.softmax(attention_scores)

        # Compute weighted sum of the values
        weighted_values = torch.matmul(attention_weights, value_proj)

        return weighted_values, attention_weights
