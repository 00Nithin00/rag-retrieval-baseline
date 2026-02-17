from __future__ import annotations

from typing import Dict, List, Any, Tuple


def rrf_fuse(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion:
    score(d) = sum_i 1 / (k + rank_i(d))
    Lower rank number = better.
    """
    scores: Dict[str, float] = {}
    best_payload: Dict[str, Dict[str, Any]] = {}

    def add(results: List[Dict[str, Any]]):
        for rank, r in enumerate(results, start=1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # keep any payload (title/text) for printing
            if cid not in best_payload:
                best_payload[cid] = r

    add(bm25_results)
    add(dense_results)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for cid, s in fused:
        r = dict(best_payload[cid])
        r["rrf_score"] = float(s)
        out.append(r)
    return out
