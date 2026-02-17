from __future__ import annotations

from typing import Dict, List, Set, Tuple


def recall_at_k(ranked_doc_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked_doc_ids[:k]
    hit = len(set(topk) & relevant)
    return hit / float(len(relevant))


def mrr_at_k(ranked_doc_ids: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked_doc_ids[:k]
    for i, doc_id in enumerate(topk, start=1):
        if doc_id in relevant:
            return 1.0 / float(i)
    return 0.0
