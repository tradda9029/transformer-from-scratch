import torch
from tokenizer import text, encode

data = torch.tensor(encode(text),dtype=torch.long)

def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    start_indices = torch.randint(0,len(data)-block_size,(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in start_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in start_indices])

    return x, y

if __name__ == "__main__":
    x,y = get_batch(data,2,4)
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)
    print(torch.equal(x[:,1:],y[:,:-1]))



