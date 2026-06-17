import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt

from config import *
from data import vocab_size, get_batch
from model import MyGPT

model = MyGPT()
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {num_params} parameters in {device} device")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

losses = []

for step in range(n_steps):
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"step {step} : loss {loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Saved trained weights to model.pth")

# Save a timestamped loss plot so previous runs are not overwritten.
os.makedirs("docs", exist_ok=True)
plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
plot_path = os.path.join("docs", f"training_loss_{attention_type}_{plot_timestamp}.png")
plt.figure()
plt.plot(losses)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"Training Loss - Attention: {attention_type}")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved training loss plot to {plot_path}")
plt.show()
