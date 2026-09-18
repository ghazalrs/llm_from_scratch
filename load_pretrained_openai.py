"""Load OpenAI GPT-2 weights into GPTModel and generate a text sample."""

import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

from gpt_model import GPTModel
from text_generation import generate, text_to_token_ids, token_ids_to_text


def assign(left, right):
    """Convert pretrained values into a parameter with the expected shape."""
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(
        torch.tensor(right, dtype=left.dtype, device=left.device)
    )


def load_weights_into_gpt(gpt, params):
    """Replace model parameters with the pretrained GPT-2 parameters."""
    if len(gpt.trf_blocks) != len(params["blocks"]):
        raise ValueError("Model and checkpoint must have the same number of blocks")
    if any(block.att.W_query.bias is None for block in gpt.trf_blocks):
        raise ValueError("GPT-2 pretrained weights require qkv_bias=True")

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block, weights in zip(gpt.trf_blocks, params["blocks"]):
        # GPT-2 stores query, key, and value projections together.
        q_w, k_w, v_w = np.split(weights["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(weights["attn"]["c_attn"]["b"], 3, axis=-1)
        for layer, weight, bias in (
            (block.att.W_query, q_w, q_b),
            (block.att.W_key, k_w, k_b),
            (block.att.W_value, v_w, v_b),
        ):
            layer.weight = assign(layer.weight, weight.T)
            layer.bias = assign(layer.bias, bias)

        # Transpose GPT-2 projection matrices for PyTorch Linear layers.
        for layer, source in (
            (block.att.out_proj, weights["attn"]["c_proj"]),
            (block.ff.layers[0], weights["mlp"]["c_fc"]),
            (block.ff.layers[2], weights["mlp"]["c_proj"]),
        ):
            layer.weight = assign(layer.weight, source["w"].T)
            layer.bias = assign(layer.bias, source["b"])

        for layer, source in (
            (block.norm1, weights["ln_1"]),
            (block.norm2, weights["ln_2"]),
        ):
            layer.scale = assign(layer.scale, source["g"])
            layer.shift = assign(layer.shift, source["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def load_pretrained_model(device="cpu", models_dir=None):
    """Load the local 124M checkpoint, downloading it if files are missing."""
    from gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt

    models_dir = (
        Path(models_dir) if models_dir is not None
        else Path(__file__).resolve().parent / "gpt2"
    )
    model_dir = models_dir / "124M"
    required_files = (
        "hparams.json", "model.ckpt.index", "model.ckpt.data-00000-of-00001"
    )
    if all((model_dir / name).is_file() for name in required_files):
        with (model_dir / "hparams.json").open(encoding="utf-8") as file:
            settings = json.load(file)
        params = load_gpt2_params_from_tf_ckpt(str(model_dir / "model.ckpt"), settings)
    else:
        settings, params = download_and_load_gpt2("124M", str(models_dir))

    config = {
        "vocab_size": settings["n_vocab"],
        "context_length": settings["n_ctx"],
        "emb_dim": settings["n_embd"],
        "n_heads": settings["n_head"],
        "n_layers": settings["n_layer"],
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    gpt = GPTModel(config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()
    return gpt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading pretrained GPT-2 124M...", flush=True)
    gpt = load_pretrained_model(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=50,
        context_size=gpt.pos_emb.num_embeddings,
        eos_id=tokenizer.eot_token,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
