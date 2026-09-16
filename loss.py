from itertools import islice

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Return mean loss per token over the selected batches."""
    if num_batches is not None and num_batches <= 0:
        raise ValueError("num_batches must be positive")

    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    total_tokens = 0
    for input_batch, target_batch in islice(data_loader, num_batches):
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        num_tokens = target_batch.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens else float("nan")
