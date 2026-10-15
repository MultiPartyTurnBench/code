"""Voice-disjoint dev/test split (paper Section 3.1 "Speaker rendering", Table 8).

The 24-voice pool is partitioned, gender-balanced, into a development pool (4 F + 4 M
voices) and a disjoint test pool (the remaining 16). Each split's conversations are then
sampled with :func:`mptb.generation.strata.sample_stratified` using only its own voices, so
operating points tuned on dev never see test-speaker acoustics.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from ..eval.schema import N_MAX


def partition_voices(voice_pool: List[Dict[str, str]], n_dev_per_gender: int = 4,
                     seed: int = 0) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Gender-balanced disjoint split of ``voice_pool`` into ``(dev_voices, test_voices)``.

    For each gender (in sorted order) the voices are shuffled with ``random.Random(seed)``
    and the first ``n_dev_per_gender`` go to dev, the rest to test. Both pools must
    contain at least ``N_MAX`` voices.
    """
    rng = random.Random(seed)
    by_g: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for v in voice_pool:
        by_g[v.get("gender", "?")].append(v)
    dev: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []
    for g in sorted(by_g):
        vs = list(by_g[g])
        rng.shuffle(vs)
        dev += vs[:n_dev_per_gender]
        test += vs[n_dev_per_gender:]
    if len(dev) < N_MAX or len(test) < N_MAX:
        raise ValueError(f"each split needs >= {N_MAX} voices "
                         f"(got dev={len(dev)}, test={len(test)}); lower n_dev_per_gender")
    return dev, test
