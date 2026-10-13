import torch
from attention_lm import AttentionLanguageModel

model_with_pos = AttentionLanguageModel(
    10, 32, 32, 8, use_positional_embedding=True
)

model_without_pos = AttentionLanguageModel(
    10,32,32,8, use_positional_embedding=False
)

idx_1 = torch.tensor([[1,2,3]])
idx_2 = torch.tensor([[2,1,3]])

model_with_pos.eval()
model_without_pos.eval()

with torch.no_grad():
    assert not torch.allclose(
        model_with_pos(idx_1)[:,-1,:],model_with_pos(idx_2)[:,-1,:]
    )

    assert torch.allclose(
        model_without_pos(idx_1)[:,-1,:],model_without_pos(idx_2)[:,-1,:]
    )
