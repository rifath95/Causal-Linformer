import torch

from config import *

# Opening the dataset and text is a long string
with open('tinyShakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extracting the characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoding & decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]
def decode(l): return ''.join(itos[i] for i in l)


# train vs val data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
