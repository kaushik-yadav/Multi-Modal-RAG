import logging
from typing import Any, Dict, List, Optional

import numpy as np

from utils.constants import DEFAULT_FETCH_K, DEFAULT_K, MMR_LAMBDA
from core.embedding import embed_texts
from core.vector_indexer import load_index_and_meta

logger = logging.getLogger(__name__)

# Optional cross-encoder, loaded lazily if available
_CROSS_ENCODER = None
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _load_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(_CE_MODEL_NAME)
        logger.info("Loaded cross-encoder %s", _CE_MODEL_NAME)
    except Exception as e:
        logger.info("Cross-encoder not available, skipping re-rank. %s", e)
        _CROSS_ENCODER = None
    return _CROSS_ENCODER

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize rows of x to unit L2 norm, safe for zero vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def _get_candidate_embeddings(candidate_items: List[Dict[str, Any]]) -> np.ndarray:
    """
    Try to read precomputed embeddings from item metadata key 'embedding'.
    If missing for any item, re-embed the missing ones and return a full array.
    """
    # Attempt to collect embeddings from metadata
    existing = []
    missing_indices = []
    for i, it in enumerate(candidate_items):
        emb = it.get("embedding")
        if emb is None:
            missing_indices.append(i)
            existing.append(None)
        else:
            # ensure numpy array float32
            existing.append(np.asarray(emb, dtype=np.float32))
    if not missing_indices:
        return np.vstack(existing).astype(np.float32)
    # Need to re-embed missing texts
    texts_to_embed = [candidate_items[i].get("content", "") for i in missing_indices]
    logger.warning("Missing %d precomputed embeddings, computing on the fly. Consider precomputing and storing embeddings to avoid this.", len(missing_indices))
    new_embs = embed_texts(texts_to_embed)
    new_embs = np.asarray(new_embs, dtype=np.float32)
    # fill into existing list
    j = 0
    for idx in missing_indices:
        existing[idx] = new_embs[j]
        j += 1
    return np.vstack(existing).astype(np.float32)

def _rerank_with_cross_encoder(query: str, candidate_texts: List[str], top_n: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Returns cross-encoder scores if a cross-encoder is available, otherwise None.
    Scores are higher = more relevant, in the same order as candidate_texts.
    Only use for small top_n lists because cross-encoders are more costly.
    """
    ce = _load_cross_encoder()
    if ce is None:
        return None
    # Prepare pairs
    pairs = [[query, t] for t in candidate_texts]
    try:
        # CrossEncoder returns a numpy array of floats
        scores = ce.predict(pairs, show_progress_bar=False, batch_size=16)
        return np.asarray(scores, dtype=np.float32)
    except Exception as e:
        logger.warning("Cross-encoder rerank failed: %s", e)
        return None

def mmr_retrieve(query: str, k: int = DEFAULT_K, fetch_k: int = DEFAULT_FETCH_K, use_cross_encoder: bool = False, cross_encoder_topk: int = 8) -> List[Dict[str, Any]]:
    """
    Vectorized MMR retrieval with optional cross-encoder re-ranking on top candidates.

    Steps:
    1) ANN search to get top fetch_k candidates
    2) load or compute candidate embeddings once
    3) compute cosine similarities between query and candidates
    4) run vectorized MMR to pick up to k diverse, relevant candidates
    5) optional small cross-encoder rerank on the final shortlist
    """
    try:
        index, items = load_index_and_meta()
        if index is None or not items:
            logger.warning("No index or items available")
            return []
        if index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # embed query
        q_embs = embed_texts([query])
        if len(q_embs) == 0:
            logger.error("Query embedding failed")
            return []
        q = np.asarray(q_embs[0], dtype=np.float32).reshape(-1)
        # normalize query
        q_norm = q / (np.linalg.norm(q) + 1e-12)

        # guard fetch_k
        fetch_k = min(fetch_k, index.ntotal)
        # initial ANN search to get candidate indices
        raw_scores, raw_indices = index.search(np.expand_dims(q.astype(np.float32), axis=0), fetch_k)
        raw_scores = raw_scores[0]
        raw_indices = raw_indices[0]

        # filter valid candidate indices and collect items
        candidate_items = []
        idx_map = []
        for idx in raw_indices:
            if idx is None:
                continue
            if int(idx) < 0:
                continue
            idx = int(idx)
            if 0 <= idx < len(items):
                candidate_items.append(items[idx])
                idx_map.append(idx)
            else:
                logger.debug("Index %s out of bounds for items length %d", idx, len(items))

        if not candidate_items:
            return []

        # load or compute embeddings for candidates, and normalize rows
        candidate_embeddings = _get_candidate_embeddings(candidate_items)  # shape (n, d)
        candidate_embeddings = candidate_embeddings.astype(np.float32)
        candidate_embeddings = _normalize_rows(candidate_embeddings)  # now cosine via dot product

        # compute cosine similarity between query and candidates
        sim_q = candidate_embeddings.dot(q_norm).astype(np.float32)  # shape (n,)

        n_candidates = candidate_embeddings.shape[0]
        mmr_lambda = float(MMR_LAMBDA)

        # MMR selection vectorized
        selected = []
        selected_mask = np.zeros(n_candidates, dtype=bool)
        # pick first as highest sim to query
        first = int(np.argmax(sim_q))
        selected.append(first)
        selected_mask[first] = True

        # precompute pairwise similarities between candidates if needed
        # For diversity computation we need candidate-to-candidate sims
        pair_sims = candidate_embeddings.dot(candidate_embeddings.T)  # shape (n, n)

        while len(selected) < min(k, n_candidates):
            # for each unselected candidate compute:
            # mmr_score = lambda * sim(query, cand) - (1 - lambda) * max(sim(cand, selected_docs))
            unselected_idx = np.where(~selected_mask)[0]
            if unselected_idx.size == 0:
                break
            sim_to_query = sim_q[unselected_idx]  # shape (m,)
            if len(selected) > 0:
                sims_to_selected = pair_sims[unselected_idx][:, selected]  # shape (m, len(selected))
                max_sim_to_selected = np.max(sims_to_selected, axis=1)
            else:
                max_sim_to_selected = np.zeros_like(sim_to_query)
            mmr_scores = mmr_lambda * sim_to_query - (1.0 - mmr_lambda) * max_sim_to_selected
            choose_pos = int(np.argmax(mmr_scores))
            choose_idx = unselected_idx[choose_pos]
            selected.append(int(choose_idx))
            selected_mask[choose_idx] = True

        # Build result list in selected order
        results = []
        for pos in selected:
            item = candidate_items[pos].copy()
            # Attach a retrieval_score that is the cosine sim to query
            item['retrieval_score'] = float(sim_q[pos])
            # map original index for traceability
            item['_index_in_store'] = int(idx_map[pos]) if pos < len(idx_map) else None
            results.append(item)

        # Optional small cross-encoder re-rank over top cross_encoder_topk of results
        if use_cross_encoder and len(results) > 0:
            ce_n = min(cross_encoder_topk, len(results))
            ce_texts = [r.get("content", "") for r in results[:ce_n]]
            ce_scores = _rerank_with_cross_encoder(query, ce_texts, top_n=ce_n)
            if ce_scores is not None:
                # replace retrieval_score for those top entries with ce score, higher is better
                for i in range(ce_n):
                    results[i]['retrieval_score'] = float(ce_scores[i])
                # finally sort results by retrieval_score descending
                results = sorted(results, key=lambda x: x['retrieval_score'], reverse=True)

        logger.info("MMR retrieval returned %d items", len(results))
        return results

    except Exception as e:
        logger.exception("MMR retrieval failed, falling back to simple retrieval: %s", e)
        return simple_retrieve(query, k)


def simple_retrieve(query: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """
    Simple ANN retrieval fallback. k is the number of results to return.
    """
    try:
        index, items = load_index_and_meta()
        if index is None or not items:
            return []

        q_embs = embed_texts([query])
        if len(q_embs) == 0:
            return []
        q = np.asarray(q_embs[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        k = min(k, index.ntotal) if index.ntotal > 0 else k
        scores, indices = index.search(np.expand_dims(q, axis=0), k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            idx = int(idx)
            if 0 <= idx < len(items):
                it = items[idx].copy()
                # use score as-is, but prefer cosine if embedding available
                if it.get("embedding") is not None:
                    cand_emb = np.asarray(it["embedding"], dtype=np.float32)
                    cand_emb = cand_emb / (np.linalg.norm(cand_emb) + 1e-12)
                    sim = float(np.dot(cand_emb, q))
                    it['retrieval_score'] = sim
                else:
                    it['retrieval_score'] = float(score)
                results.append(it)
        return results
    except Exception as e:
        logger.exception("Simple retrieval failed: %s", e)
        return []
