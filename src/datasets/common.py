"""Shared deterministic sampling helpers."""

from __future__ import annotations

import hashlib
import random
from typing import Callable, List, Sequence, TypeVar


T = TypeVar("T")


def dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(("%d:%s" % (seed, dataset)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_sample(
    values: Sequence[T],
    count: int,
    *,
    seed: int,
    dataset: str,
    unique_key: Callable[[T], str],
) -> List[T]:
    unique = {}
    for value in values:
        key = " ".join(unique_key(value).casefold().split())
        if key and key not in unique:
            unique[key] = value
    candidates = list(unique.values())
    if len(candidates) < count:
        raise ValueError(
            "%s has only %d unique valid cases; %d requested"
            % (dataset, len(candidates), count)
        )
    rng = random.Random(dataset_seed(seed, dataset))
    return rng.sample(candidates, count)

