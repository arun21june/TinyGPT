import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head

        self.head_dim = d_model // num_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
  
    def forward(self, X):

        batch, seq_len, _ = X.shape

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # split into heads
        Q = Q.view(batch, seq_len, self.num_head, self.head_dim)
        K = K.view(batch, seq_len, self.num_head, self.head_dim)
        V = V.view(batch, seq_len, self.num_head, self.head_dim)
        
        # put head before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device))

        # Attention
        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim= -1)

        context = weights @ V 

        # Combine heads
        context = context.transpose(1,2)
        context = context.reshape(batch, seq_len, self.d_model)

        #Final projection
        output = self.out_proj(context)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        X = self.fc1(X)
        X = F.gelu(X)
        X = self.fc2(X)
        return X

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiheadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):

        # 1. self-attention
        attention_output = self.attention(X)

        # 2. Risidual connection
        X = X + attention_output

        # 3. LayerNorm
        X = self.norm1(X)

        # 4. FFN
        ffn_output = self.ffn(X)

        #5. Residual connection
        X = X + ffn_output

        #6. LayerNorm
        X = self.norm2(X)

        return X





        

