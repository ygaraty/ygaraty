"""Stable operation labels shared by measured and simulated reports."""

FAMILIES = {
    "lstm_cell": "LSTM forward steps",
    "lstm_cell_grad": "LSTM backward steps",
    "lstm_state_grad": "LSTM backward steps",
    "tf_layer": "Transformer layers (forward)",
    "tf_layer_grad": "Transformer layers (backward)",
    "tf_owned_layer": "Transformer layers (forward)",
    "tf_owned_grad": "Transformer layers (backward)",
    "tf_owned_step": "optimizer",
    "tf_owned_load": "owned layer load and publish",
    "tf_owned_publish": "owned layer load and publish",
    "tf_vocab_load": "owned layer load and publish",
    "tf_vocab_publish": "owned layer load and publish",
    "tf_vocab_step": "optimizer",
    "tf_vocab_gather": "owned vocabulary (embedding rows)",
    "tf_vocab_embed_grad": "owned vocabulary (embedding rows)",
    "tf_vocab_stats": "output softmax and loss",
    "vocab_loss": "output softmax and loss",
    "tf_vocab_grad": "output gradient (owned vocabulary)",
    "ce_fused": "output softmax and loss",
    "layer_norm": "layer norms",
    "layer_norm_backward": "layer norms",
    "lstm_chain": "LSTM forward steps",
    "lstm_chain_grad": "LSTM backward steps",
    "matmul": "matmuls (projections, weight gradients)",
    "add_many": "sums of partial products",
    "reduce_rows": "bias gradients (reduce_rows)",
    "scatter_rows": "embedding gradient (scatter_rows)",
    "place_block": "placements and concatenations",
    "add": "gradient accumulation (add)",
    "bqk": "attention",
    "bpv": "attention",
    "bptq": "attention",
    "softmax": "attention",
    "softmax_backward": "attention",
    "cross_entropy": "output softmax and loss",
    "ce_backward": "output softmax and loss",
    "assemble": "metadata (coordinator)",
    "transpose": "metadata (coordinator)",
}


def family(op, task=""):
    if op.startswith("adam") or op == "sgd":
        return "optimizer"
    if op == "matmul" and (
        "/hw/" in task or "/dh/" in task or "/dh0/" in task
    ):
        return "recurrent products (chain)"
    return FAMILIES.get(op, "elementwise and other")
