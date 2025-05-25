import math
import os
from tqdm import tqdm
import torch


def get_random_batch(tokens, context_size, batch_size):
    ix = torch.randint(len(tokens) - context_size - 1 , (batch_size,))
    x = torch.stack([tokens[i:i+context_size] for i in ix])
    y = torch.stack([tokens[i+1:i+context_size+1] for i in ix])
    return x, y

def tokenize_files(files, encode, eod_token):
    tokens = []
    texts = []
    for file in tqdm(files):
        with open(file, "r", encoding='utf-8') as f:
            text = f.read()
        text_encoded = encode(text)
        
        tokens += text_encoded
        tokens += encode(eod_token)

        texts += text
        texts += eod_token
    return tokens, texts

def get_all_files(directory):
    all_files = []
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        # Add the full path of each file to the list
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return all_files


# Cosine learning rate scheduler with warmup
def get_lr(step, total_steps, lr_max, lr_min, warmup_steps, num_of_steps):
    # Warmup phase
    if step < warmup_steps:
        return lr_max * float(step) / float(max(1, warmup_steps))
    if step > num_of_steps:
        return lr_min

    # Cosine decay phase
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    """
    Save model checkpoint for continued training

    Args:
        model: Your PyTorch model
        optimizer: Your optimizer instance
        epoch: Current epoch number
        loss: Current loss value
        filename: Path where to save the checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True
    )

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    """
    Load model checkpoint to resume training

    Args:
        model: Your PyTorch model
        optimizer: Your optimizer instance
        filename: Path to the checkpoint file

    Returns:
        epoch: The epoch number where training left off
        loss: The loss value at the time of saving
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return 0, None

    # Load checkpoint
    checkpoint = torch.load(filename)

    # Restore state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss


# Example usage in your training loop:
"""
# Initialize model and optimizer
model = AttentionModel(vocab_size, att_size, head_count, layer_count, CONTEXT_SIZE, DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# If you want to resume training, load checkpoint
start_epoch, last_loss = load_checkpoint(model, optimizer, "checkpoints/model.pt")

# Training loop
for epoch in range(start_epoch, num_epochs):
    # ... your training code ...
    
    # Save checkpoint periodically
    if epoch % save_every == 0:
        save_checkpoint(model, optimizer, epoch, current_loss, f"checkpoints/model_epoch_{epoch}.pt")
    
    # Also save on the last epoch
    if epoch == num_epochs - 1:
        save_checkpoint(model, optimizer, epoch, current_loss, "checkpoints/model_final.pt")
"""


# If you want to just save the model weights only (without optimizer state)
def save_model_only(model, filename="model.pt"):
    """Save just the model weights"""
    torch.save(model.state_dict(), filename)
    print(f"Model weights saved to {filename}")


def load_model_only(model, filename="model.pt"):
    """Load just the model weights"""
    model.load_state_dict(torch.load(filename))
    print(f"Model weights loaded from {filename}")
    return model
