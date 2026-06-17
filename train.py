import os  # [Codex comment] Create the plot output folder and build a portable save path.
from datetime import datetime  # [Codex comment] Generate a unique timestamp for every saved training plot.
import torch  # [Codex comment] Keep PyTorch available for model training and checkpoint saving.
import matplotlib.pyplot as plt

from config import *
from data import vocab_size, get_batch
from model import TestGPT

model = TestGPT()
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

# [Codex comment] Plot and save the training loss with labels that identify the attention implementation.
os.makedirs("docs", exist_ok=True)  # [Codex comment] Ensure the plot destination exists before saving the image.
plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # [Codex comment] Include microseconds so repeated runs cannot receive the same plot name.
plot_path = os.path.join("docs", f"training_loss_{attention_type}_{plot_timestamp}.png")  # [Codex comment] Save every attention run under a unique filename without overwriting earlier plots.
plt.figure()  # [Codex comment] Create a fresh figure for this training run.
plt.plot(losses)  # [Codex comment] Plot recorded loss values in training-step order.
plt.xlabel("Training Step")  # [Codex comment] Label the horizontal axis with the optimization-step meaning.
plt.ylabel("Cross-Entropy Loss")  # [Codex comment] Label the vertical axis with the model's training objective.
plt.title(f"Training Loss - Attention: {attention_type}")  # [Codex comment] Include the configured attention type in the plot title.
plt.grid(True)  # [Codex comment] Add a grid to make loss values easier to compare visually.
plt.tight_layout()  # [Codex comment] Prevent the title and axis labels from being clipped in the saved image.
plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # [Codex comment] Save a high-resolution labeled loss plot in the docs folder.
print(f"Saved training loss plot to {plot_path}")  # [Codex comment] Report the exact output path after saving.
plt.show()  # [Codex comment] Preserve the existing behavior of displaying the plot after training.
