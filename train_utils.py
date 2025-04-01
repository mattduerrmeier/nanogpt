from torch import optim
from model import GPT
import math


def get_lr(step: int) -> float:
    MAX_LR = 6e-4
    MIN_LR = MAX_LR * 0.1
    WARMUP_STEPS = 10
    MAX_STEPS = 50
    # warmup and after warmup regions
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    elif step > WARMUP_STEPS:
        return MIN_LR

    # in between: decrease the LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def setup_optimizer(model: GPT) -> optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    return optimizer
