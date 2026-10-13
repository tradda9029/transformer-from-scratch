import torch
from bigram import BigramLanguageModel
from attention_lm import AttentionLanguageModel

V = 10
C = D = 32
T = 8

model_Bigram = BigramLanguageModel(V)
model_Attention_without_position = AttentionLanguageModel(V,C,D,T,False)
model_Attention_with_position = AttentionLanguageModel(V,C,D,T)

result_Bigram = sum(param.numel() for param in model_Bigram.parameters())
result_Attention_without_position = sum(param.numel() for param in model_Attention_without_position.parameters())
result_Attention_with_position = sum(param.numel() for param in model_Attention_with_position.parameters())

print("The parameters of Bigram model is ",result_Bigram )
print("The parameters of Attention model without position embedding ",result_Attention_without_position)
print("The parameters of Attention model with position embedding ",result_Attention_with_position)