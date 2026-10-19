"""Tensor parallel (TP=2) config handling - the TP-owned module (modularization: server.py keeps exactly
three one-line seams into here, so rebasing onto a future upstream release touches TP logic in one file).

The engine is the authority on what TP accepts (`program/tp_rules.hpp`); what is checked here is only what
would otherwise cost a model load before the same complaint.
"""


def tp_of(cfg: dict) -> bool:
    """The config asks for TENSOR parallelism: `"tp": 2`, `"tp": true`, or `"tp": [0, 2]` (a list names the CARDS,
    the way "gpu" does).  TP and the layer split are two different ways to spend a second card and the engine
    refuses both at once (`program/tp_rules.hpp`), so a config that asks for TP deliberately does NOT get
    --layer-split.  Off: absent, false, 0, or an empty list."""
    v = cfg.get("tp")
    if v is None:
        return False
    # `True` and `2` both mean "on" (the engine's value is always 2), and `isinstance(True, int)` is True, so the
    # integer branch covers the boolean spelling as well
    if isinstance(v, int):
        return v > 0
    return isinstance(v, (list, tuple, str)) and len(v) > 0


def tp_value(cfg: dict) -> str:
    """The engine's --tp value.  The engine has exactly one (2: `n_head_kv` is 2, so a third rank would have
    no KV head of its own), so a list in the config names the CARDS rather than the rank count."""
    return "2"


def tp_extra_args(cfg: dict, args: list) -> list:
    """The engine args TP adds: `--tp 2`, once.  Both ranks run every layer and split each dense tensor, so
    the cards come from "gpu" as they do for a layer split (and `layer_split_of` is false when TP is on, so
    nothing else is added)."""
    if tp_of(cfg) and "--tp" not in args:
        return ["--tp", tp_value(cfg)]
    return []


def tp_startup_check(cfg: dict, gpu_list, engine_args) -> None:
    """The checks main() runs before starting the engine, so a wrong config fails without costing a model
    load.  Only --layer-split is refused here (and by the engine): it is the other way to spend a second
    card.  Everything else - --vision, --kv-resident, --mmap-experts, --batch, --pipeline-windows,
    --peer-device, the mid-prompt cache - is NOT refused: both ranks run the same window on the same
    tokens, so images go through the same vision path and each rank streams its own share of the KV.
    `program/tp_rules.hpp` is the authority and refuses none of it ("llama.cpp's tensor parallelism refuses
    none of it, and neither does this one"); this check only saves a model load before the engine's own
    complaint.  Both ranks must run the SAME reduces in the SAME order, so every flag that makes one
    rank's window differ is refused by the engine."""
    cards = gpu_list(cfg)
    if len(cards) != 2:
        raise SystemExit(f'[strata] config: "tp" needs exactly two cards, and "gpu" names {len(cards)}')
    eargs = engine_args(cfg)
    for bad, why in (("--layer-split", "it is the other way to use a second card"),):
        if bad in eargs:
            raise SystemExit(f'[strata] config: "tp" cannot be combined with {bad} ({why})')
    print(f"[strata] tensor parallel across GPUs {cards} (both cards run every layer; "
          "see docs/MULTI_GPU.md)", flush=True)
