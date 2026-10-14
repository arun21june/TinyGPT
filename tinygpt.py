from transformerBlock import TransformerBlock
import torch
from torch import nn


class TinyGPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            block_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias = False
        )

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape

        # Token Identity 
        token_embeddings = self.embedding(tokens)

        # Token Position
        positions = torch.arange(seq_len, device=tokens.device)
        position_embeddings = self.position_embedding(positions)

        # Combine Token identity and position
        X= token_embeddings + position_embeddings

        for block in self.blocks:
            X = block(X)

        X = self.norm(X)

        logits = self.lm_head(X)

        return logits
