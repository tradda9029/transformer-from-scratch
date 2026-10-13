from src.self_attention import SelfAttention
import torch
import torch.nn as nn


class AttentionLanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            head_dim: int,
            block_size: int,
            use_positional_embedding: bool = True
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.attention = SelfAttention(embedding_dim,head_dim)
        self.lm_head = nn.Linear(head_dim,vocab_size)

        if use_positional_embedding:
            self.position_embedding = nn.Embedding(block_size,embedding_dim)
        else:
            self.position_embedding = None

        self.use_positional_embedding = use_positional_embedding
        self.block_size = block_size

    def forward(self, idx: torch.Tensor):
        sequence_length = idx.shape[1]

        if sequence_length > self.block_size :
            raise ValueError("sequence length is longer than block size")

        x = self.token_embedding(idx)

        if self.use_positional_embedding:
            positions_id = torch.arange(sequence_length, device=idx.device)
            position_embedded = self.position_embedding(positions_id)
            x = x + position_embedded

        z = self.attention(x)
        logits = self.lm_head(z)
        return logits

    def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)

            last_logits = logits[:,-1,:]

            probs = torch.softmax(last_logits,dim=-1)

            idx_next = torch.multinomial(probs,num_samples=1)

            idx = torch.cat([idx,idx_next],dim=1)

        return idx
