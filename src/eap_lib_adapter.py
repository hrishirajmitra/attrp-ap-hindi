"""Optional adapter to hannamw/EAP-IG for cross-checking our AttrP rankings.

Install on Ada:
  pip install "git+https://github.com/hannamw/EAP-IG.git"

Primary engine remains our AP/AttrP (see results/ENGINE_DECISION.md).

The library expects model.cfg flags:
  use_attn_result, use_split_qkv_input, use_hook_mlp_in
and for GQA: ungroup_grouped_query_attention=True.
Dataloader batches are (clean_str_list, corrupted_str_list, label).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .batch import TaskBatch

_HEAD_RE = re.compile(r"^a(\d+)\.h(\d+)$")
_MLP_RE = re.compile(r"^m(\d+)$")


def eap_available() -> bool:
    try:
        import eap  # noqa: F401

        return True
    except ImportError:
        return False


def configure_model_for_eap(model) -> None:
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if getattr(model.cfg, "n_key_value_heads", None) is not None:
        model.cfg.ungroup_grouped_query_attention = True


def _string_dataloader(batch: TaskBatch, batch_size: int = 8):
    from torch.utils.data import DataLoader

    clean = batch.prompts_clean
    corr = batch.prompts_corrupted
    # Dummy labels (metric ignores them when we wrap)
    labels = list(range(len(clean)))

    def collate(rows):
        cs, rs, ls = zip(*rows)
        return list(cs), list(rs), list(ls)

    pairs = list(zip(clean, corr, labels))
    return DataLoader(pairs, batch_size=batch_size, collate_fn=collate)


def node_attrp_from_eap_lib(
    model,
    batch: TaskBatch,
    method: str = "EAP",
    ig_steps: int = 5,
    batch_size: int = 8,
    quiet: bool = True,
) -> dict[str, Any]:
    """Run node-level EAP / EAP-IG-inputs and reshape to [L,H] / [L]."""
    if not eap_available():
        raise ImportError(
            "eap package not installed. "
            'pip install "git+https://github.com/hannamw/EAP-IG.git"'
        )

    from eap.attribute_node import attribute_node
    from eap.graph import Graph

    from .metrics import logit_diff_from_logits

    configure_model_for_eap(model)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    graph = Graph.from_model(model)

    # Metric signature in EAP-IG: metric(logits, clean_logits, input_lengths, label)
    def metric(logits, clean_logits=None, input_lengths=None, label=None):
        _ = clean_logits, input_lengths, label
        # Align batch size with current forward (may be a microbatch)
        b = logits.shape[0]
        return logit_diff_from_logits(
            logits, batch.pos_token_ids[:b], batch.neg_token_ids[:b]
        ).mean()

    method_map = {
        "EAP": "EAP",
        "AttrP": "EAP",
        "EAP-IG": "EAP-IG-inputs",
        "EAP-IG-inputs": "EAP-IG-inputs",
    }
    method_key = method_map.get(method, method)

    attribute_node(
        model,
        graph,
        _string_dataloader(batch, batch_size=batch_size),
        metric,
        method=method_key,
        ig_steps=ig_steps if "IG" in method_key else None,
        quiet=quiet,
    )

    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    mlp_scores = np.zeros(n_layers, dtype=np.float64)
    for name, node in graph.nodes.items():
        score = float(getattr(node, "score", 0.0) or 0.0)
        m = _HEAD_RE.match(name)
        if m:
            layer, head = int(m.group(1)), int(m.group(2))
            if layer < n_layers and head < n_heads:
                head_scores[layer, head] = score
            continue
        m = _MLP_RE.match(name)
        if m:
            layer = int(m.group(1))
            if layer < n_layers:
                mlp_scores[layer] = score

    return {
        "method": f"eap_lib_{method_key}",
        "head_scores": head_scores,
        "mlp_scores": mlp_scores,
        "clean_logit_diff": None,
        "corrupted_logit_diff": None,
        "denom": None,
        "wall_seconds": None,
        "n_prompts": int(batch.clean_tokens.shape[0]),
        "note": "Cross-check only (hannamw/EAP-IG node scores)",
    }
