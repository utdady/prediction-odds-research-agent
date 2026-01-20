from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationResult:
    p_norm: float
    yes_mid: float | None
    no_mid: float | None
    illiquid_flag: bool


def clip_prob(p: float) -> float:
    return min(0.999, max(0.001, p))


def normalize_yes_no(yes: float | None, no: float | None) -> NormalizationResult:
    if yes is None and no is None:
        raise ValueError("missing yes/no")
    illiquid = False

    if yes is None and no is not None:
        yes = 1.0 - no
    if no is None and yes is not None:
        no = 1.0 - yes

    assert yes is not None and no is not None
    s = yes + no
    if abs(s - 1.0) > 0.02:
        illiquid = True
    return NormalizationResult(p_norm=clip_prob(yes), yes_mid=yes, no_mid=no, illiquid_flag=illiquid)

