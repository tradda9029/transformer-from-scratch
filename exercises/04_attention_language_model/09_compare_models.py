import torch
import torch.nn as nn
from attention_lm import AttentionLanguageModel
from bigram import BigramLanguageModel
from dataset import data, get_batch
from tokenizer import vocab_size

batch_size = 8
block_size = 8
embedding_dim = 32
head_dim = 32
max_steps = 1000
learning_rate = 0.001

def train_model(model, train_batches):
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history = []

    for step, (x,y) in enumerate(train_batches):
        logits = model(x)
        logits_flat = logits.reshape(-1,logits.shape[-1])
        targets_flat = y.reshape(-1,)

        loss = loss_fn(logits_flat,targets_flat)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            loss_history.append((step, loss.item()))

    return loss_history

def evaluate_model(model, eval_x, eval_y) -> float:
    model.eval()

    with torch.no_grad():
        logits = model(eval_x)
        logits_flat = logits.reshape(-1,logits.shape[-1])
        targets_flat = eval_y.reshape(-1,)

        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits_flat,targets_flat)

    return loss.item()

if __name__ == "__main__":
    torch.manual_seed(0)

    train_batches = [
        get_batch(data, batch_size, block_size)
        for _ in range(max_steps)
    ]

    model_bigram = BigramLanguageModel(vocab_size)

    torch.manual_seed(42)
    model_without_pos = AttentionLanguageModel(vocab_size,embedding_dim,head_dim,block_size,False)

    torch.manual_seed(42)
    model_with_pos = AttentionLanguageModel(vocab_size,embedding_dim,head_dim,block_size)

    history_bigram = train_model(model_bigram,train_batches)
    history_without_pos = train_model(model_without_pos,train_batches)
    history_with_pos = train_model(model_with_pos,train_batches)

    print(history_bigram)
    print(history_without_pos)
    print(history_with_pos)


    eval_x = torch.stack([data[i:i+block_size] for i in range(len(data)-block_size)])
    eval_y = torch.stack([data[i+1:i+block_size+1] for i in range(len(data)-block_size)])

    print(evaluate_model(model_bigram,eval_x,eval_y))
    print(evaluate_model(model_without_pos,eval_x,eval_y))
    print(evaluate_model(model_with_pos,eval_x,eval_y))