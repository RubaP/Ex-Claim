import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        out = self.fc2(hidden)
        return out


class ScaledAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1./(hidden_dim ** 0.5)

    def forward(self, query, key, value):
        # query, key, value - (batch_size, seq_length, hidden_dim)
        scores = torch.bmm(query, key.transpose(1,2))  # (batch_size, seq_length, seq_length)
        scores = scores * self.scale
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)

        # Apply attention weights
        context_vector = torch.bmm(attention_weights, value)  # (batch_size, seq_length, hidden_dim)
        return context_vector, attention_weights


class XClaim(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XClaim, self).__init__()
        self.hidden_size = hidden_size
        self.attention = ScaledAttention(hidden_size)
        self.projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)
        
    def forward(self, x):
        projected_x = self.projection(x)
        context_vector, attention_weights = self.attention(projected_x, projected_x, projected_x)

        # Aggregate context vectors by taking the mean across the same sequence length
        context_vector = context_vector.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.fc(context_vector)  # (batch_size, num_classes)
        return logits


class EXClaim(nn.Module):
    def __init__(self, we_size, ee_size, no_entity, hidden_size, output_size):
        super(EXClaim, self).__init__()
        self.hidden_size = hidden_size
        self.we_size = we_size
        self.ee_size = ee_size
        self.no_entity = no_entity
        self.entity_embedding = nn.Embedding(no_entity, ee_size)
        self.attention = ScaledAttention(hidden_size)
        self.project1 = nn.Linear(we_size, hidden_size, bias=False)
        self.project2 = nn.Linear(ee_size, hidden_size, bias=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, we, e):
        ee = self.entity_embedding(e)
        w_projected = self.project1(we)
        e_projected = self.project2(ee)
        x = w_projected + e_projected
        context_vector, attention_weights = self.attention(x, x, x)

        # Aggregate context vectors by taking the mean across the same sequence length
        context_vector = context_vector.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.fc(context_vector)  # (batch_size, num_classes)
        return logits
