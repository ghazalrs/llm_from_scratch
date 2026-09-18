import argparse
from pathlib import Path

import tiktoken
import torch

from gpt_model import GPTModel
from loss import calc_loss_loader
from data_loader import create_dataloader
from pretraining import train_model_simple


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}


def main(num_epochs=1, resume=False, checkpoint_path=None):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None
        else Path(__file__).resolve().parent / "model_and_optimizer.pth"
    )
    torch.manual_seed(123)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Keep the existing PyTorch version requirement for MPS.
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()  # Disable dropout during inference

    file_path = Path(__file__).resolve().parent / "the-verdict.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    with torch.no_grad():  # Evaluation does not require gradients.
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT and save a checkpoint.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run")
    parser.add_argument("--resume", action="store_true", help="Load the checkpoint before training")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load and save")
    args = parser.parse_args()
    main(num_epochs=args.epochs, resume=args.resume, checkpoint_path=args.checkpoint)
