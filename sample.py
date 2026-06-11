import torch

from config import *
from data import encode, decode
from model import TestGPT

model = TestGPT()
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

prompt = "Hey there fellow"
context = torch.tensor(encode(prompt), device=device).unsqueeze(0)

generated_context = model.generate(context, max_new_tokens=1000)
generated_text = decode(generated_context[0].tolist())

print(generated_text)
