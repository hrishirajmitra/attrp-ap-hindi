"""Literature IOI head inventory for GPT-2 small (Wang et al. / Conmy et al.)."""

from __future__ import annotations

# (layer, head) pairs — 0-indexed
NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]
S_INHIBITION = [(7, 3), (7, 9), (8, 6), (8, 10)]
INDUCTION = [(5, 5), (5, 8), (6, 9)]
DUPLICATE_TOKEN = [(0, 1), (3, 0)]
PREVIOUS_TOKEN = [(2, 2), (4, 11)]

POSITIVE_HEADS: list[tuple[int, int]] = sorted(
    set(NAME_MOVERS + S_INHIBITION + INDUCTION + DUPLICATE_TOKEN + PREVIOUS_TOKEN)
)

HEAD_ROLE: dict[tuple[int, int], str] = {}
for h in NAME_MOVERS:
    HEAD_ROLE[h] = "name_mover"
for h in S_INHIBITION:
    HEAD_ROLE[h] = "s_inhibition"
for h in INDUCTION:
    HEAD_ROLE[h] = "induction"
for h in DUPLICATE_TOKEN:
    HEAD_ROLE[h] = "duplicate_token"
for h in PREVIOUS_TOKEN:
    HEAD_ROLE[h] = "previous_token"


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def positive_labels() -> set[str]:
    return {head_label(l, h) for l, h in POSITIVE_HEADS}
