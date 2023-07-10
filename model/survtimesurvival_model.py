import torch

from torch import nn
from torch.nn import functional as F


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, masks):
        attn_output, attn_output_weights = self.self_attn(x, x, x, key_padding_mask=(1 - masks).bool())
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_length, embed_dim, num_heads, ffn_hidden_dim, num_layers):
        # num_classes -> push if we want to define to classification
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self.create_positional_encoding(seq_length, embed_dim)
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_layer = nn.Linear(embed_dim, 21)
        # self.classifier = nn.Linear(21, num_classes)

    def create_positional_encoding(self, seq_length, embed_dim):
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_enc = torch.zeros(seq_length, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x, masks):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x, masks)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.feature_layer(x)
        return x


class TransformerClassifierFirstSolution(nn.Module):
    def __init__(self, input_dim, seq_length, embed_dim, num_heads, ffn_hidden_dim, num_layers, num_classes):
        # num_classes -> push if we want to define to classification
        super(TransformerClassifierFirstSolution, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self.create_positional_encoding(seq_length, embed_dim)
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_layer = nn.Linear(embed_dim, 21)
        self.classifier = nn.Linear(21, num_classes)

    def create_positional_encoding(self, seq_length, embed_dim):
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_enc = torch.zeros(seq_length, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x, masks):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x, masks)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.feature_layer(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class TransformerClassifier_2(nn.Module):
    def __init__(self, input_dim, seq_length, embed_dim, num_heads, ffn_hidden_dim, num_layers, num_classes):
        super(TransformerClassifier_2, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self.create_positional_encoding(seq_length, embed_dim)
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_layer = nn.Linear(embed_dim, 21)
        self.classifier = nn.Linear(21, num_classes)

    def create_positional_encoding(self, seq_length, embed_dim):
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_enc = torch.zeros(seq_length, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x, masks):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x, masks)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.feature_layer(x)
        # x = self.classifier(x)
        return x
