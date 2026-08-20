"""
Unified metric module for the ECCV-2026 benchmark suite.

Every baseline (Florence-2 unified, disjoint cascade, Molmo) and every
ablation variant runs through this single module. This guarantees the
rebuttal table is apples-to-apples.

Inputs are two parallel lists of dicts in the unified schema (see
README.md). Outputs are a flat dict of metric values + their breakdowns.

Implemented metrics:
    AP@50                 - player detection (single-class).
    ocr_num_exact         - jersey-number exact-match precision/recall/F1.
    ocr_text_fuzzy        - normalized Levenshtein >= 0.7 (rev: free-form OCR).
    team_cluster          - Hungarian-matched cluster purity vs GT teams.
    sport_type_accuracy   - frame-level exact-match accuracy of the
                            ``sport_type`` head (ECCV-2026 schema).
    scene_sbert_cosine    - Sentence-BERT cosine sim of scene_description.
    latency_ms_mean       - end-to-end ms/frame averaged across the split.

The Sentence-BERT model loads lazily; if `sentence-transformers` is not
installed, the metric returns None and the rest of the suite still runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

import Levenshtein
from scipy.optimize import linear_sum_assignment

from .ocr_text import normalize_ocr_text


IOU_MATCH_THRESHOLD = 0.5
OCR_FUZZY_THRESHOLD = 0.7

_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_sbert_model = None    # lazy singleton


# =========================================================================
# Bbox helpers (inputs are normalized [x1, y1, x2, y2] in 0..1)
# =========================================================================

def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# =========================================================================
# Greedy IoU matching (predicted -> GT)
# =========================================================================

@dataclass
class FrameMatches:
    """Result of IoU-matching predicted players to GT players for one frame."""
    pred_to_gt: Dict[int, int] = field(default_factory=dict)  # pred_idx -> gt_idx
    matched_iou: Dict[int, float] = field(default_factory=dict)
    unmatched_preds: List[int] = field(default_factory=list)
    unmatched_gts: List[int] = field(default_factory=list)


def _match_predictions(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> FrameMatches:
    """Greedy by descending pred-confidence, then by IoU."""
    pred_idx_sorted = sorted(
        range(len(preds)),
        key=lambda i: -float(preds[i].get("confidence", 1.0)),
    )
    used_gt: set = set()
    out = FrameMatches(unmatched_gts=list(range(len(gts))))

    for pi in pred_idx_sorted:
        pred_bbox = preds[pi].get("bbox") or []
        best_iou = 0.0
        best_gj = -1
        for gj in range(len(gts)):
            if gj in used_gt:
                continue
            iou = _bbox_iou(pred_bbox, gts[gj].get("bbox") or [])
            if iou > best_iou:
                best_iou = iou
                best_gj = gj
        if best_gj >= 0 and best_iou >= iou_threshold:
            out.pred_to_gt[pi] = best_gj
            out.matched_iou[pi] = best_iou
            used_gt.add(best_gj)
        else:
            out.unmatched_preds.append(pi)
    out.unmatched_gts = [g for g in out.unmatched_gts if g not in used_gt]
    return out


# =========================================================================
# AP@50 (single-class player detection)
# =========================================================================

def average_precision_at_50(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> Dict[str, float]:
    """Compute single-class AP at IoU >= ``iou_threshold`` (default 0.5).

    Implements the standard 11-point interpolated average precision over
    a sorted-by-confidence list of (TP, FP) flags.
    """
    all_records: List[Tuple[float, int, int]] = []   # (conf, tp, fp)
    total_gts = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_players = pred.get("players", []) or []
        gt_players = gt.get("players", []) or []
        total_gts += len(gt_players)

        if not pred_players:
            continue

        matches = _match_predictions(pred_players, gt_players, iou_threshold)
        for pi, p in enumerate(pred_players):
            conf = float(p.get("confidence", 1.0))
            if pi in matches.pred_to_gt:
                all_records.append((conf, 1, 0))
            else:
                all_records.append((conf, 0, 1))

    if total_gts == 0:
        return {"AP50": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}

    all_records.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in all_records])
    fps = np.cumsum([r[2] for r in all_records])
    recall = tps / float(total_gts)
    precision = tps / np.maximum(tps + fps, 1)

    # 11-point interpolation (matches PASCAL VOC).
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recall >= t
        p = float(precision[mask].max()) if mask.any() else 0.0
        ap += p / 11.0

    return {
        "AP50": float(ap),
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "tp": int(tps[-1]) if len(tps) else 0,
        "fp": int(fps[-1]) if len(fps) else 0,
        "fn": int(total_gts - (tps[-1] if len(tps) else 0)),
    }


# =========================================================================
# OCR metrics
# =========================================================================

def _normalized_levenshtein(a: str, b: str) -> float:
    """Return distance / max(len). 0.0 = identical, 1.0 = totally different."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def ocr_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    fuzzy_threshold: float = OCR_FUZZY_THRESHOLD,
) -> Dict[str, float]:
    """Compute jersey-number exact-match + free-form OCR fuzzy metrics.

    - Predictions and GTs are matched by bbox IoU first; only the matched
      pair contributes to the OCR scores (mirrors the paper's
      "association-gated OCR" semantics).
    - ``ocr_num`` counts only digit-only GT jerseys. ``ocr_text`` counts
      any non-empty GT OCR string.
    """
    num_gt_total = 0
    num_pred_total = 0
    num_correct = 0

    text_gt_total = 0
    text_pred_total = 0
    text_fuzzy_hits = 0
    text_fuzzy_score = 0.0     # accumulated 1 - normalized_levenshtein

    for pred, gt in zip(predictions, ground_truths):
        pred_players = pred.get("players", []) or []
        gt_players = gt.get("players", []) or []
        if not gt_players:
            continue
        matches = _match_predictions(pred_players, gt_players, iou_threshold)

        # Count GT supply. Normalize first so scoring space is visible-chars on
        # BOTH sides (defensive: callers that skip eval_sports._gt_view still get
        # '?'-repaired jerseys; normalize_ocr_text is idempotent on clean text).
        for gp in gt_players:
            jg = normalize_ocr_text(gp.get("jersey_number") or "")
            if jg.isdigit():
                num_gt_total += 1
            if jg:
                text_gt_total += 1

        # Score matched predictions.
        for pi, gj in matches.pred_to_gt.items():
            pp = pred_players[pi]
            gp = gt_players[gj]
            jp = normalize_ocr_text(pp.get("jersey_number") or "")
            jg = normalize_ocr_text(gp.get("jersey_number") or "")

            if jp.isdigit():
                num_pred_total += 1
                if jg.isdigit() and jp == jg:
                    num_correct += 1

            if jp:
                text_pred_total += 1
                if jg:
                    distance = _normalized_levenshtein(jp.lower(), jg.lower())
                    similarity = 1.0 - distance
                    text_fuzzy_score += similarity
                    if similarity >= fuzzy_threshold:
                        text_fuzzy_hits += 1

    def _safe(num: int, denom: int) -> float:
        return float(num) / float(denom) if denom > 0 else 0.0

    num_precision = _safe(num_correct, num_pred_total)
    num_recall = _safe(num_correct, num_gt_total)
    num_f1 = _safe(
        2 * num_precision * num_recall,
        max(num_precision + num_recall, 1e-9),
    )

    text_precision = _safe(text_fuzzy_hits, text_pred_total)
    text_recall = _safe(text_fuzzy_hits, text_gt_total)
    text_f1 = _safe(
        2 * text_precision * text_recall,
        max(text_precision + text_recall, 1e-9),
    )
    text_avg_sim = _safe(
        int(text_fuzzy_score * 1_000_000),
        text_pred_total * 1_000_000,
    )

    return {
        "ocr_num_precision": num_precision,
        "ocr_num_recall": num_recall,
        "ocr_num_f1": num_f1,
        "ocr_num_gt": num_gt_total,
        "ocr_num_pred": num_pred_total,
        "ocr_num_correct": num_correct,
        "ocr_text_precision": text_precision,
        "ocr_text_recall": text_recall,
        "ocr_text_f1": text_f1,
        "ocr_text_avg_similarity": text_avg_sim,
        "ocr_text_gt": text_gt_total,
        "ocr_text_pred": text_pred_total,
        "ocr_text_fuzzy_hits": text_fuzzy_hits,
    }


# =========================================================================
# Multi-OCR set F1 (rich cycle-3 ``ocr_items`` lists)
# =========================================================================

def _quad_pairs_to_aabb(pairs: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Axis-aligned bounding box of a 4-point quad (relative coords)."""
    if not pairs or len(pairs) < 3:
        return None
    xs = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _ocr_items_for_player(player: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the rich ocr_items list, falling back to a single-item view.

    OCR texts are normalized (visible-chars space) and items that become empty
    (all-'?') are dropped, so counts + matching both operate on repaired text.
    """
    normalized: List[Dict[str, Any]] = []
    for it in (player.get("ocr_items") or []):
        if not isinstance(it, dict):
            continue
        t = normalize_ocr_text(it.get("text") or "")
        if not t:
            continue
        normalized.append({**it, "text": t})
    if normalized:
        return normalized
    quad = player.get("ocr_quad")
    text = normalize_ocr_text(player.get("jersey_number") or "")
    if quad and text:
        return [{"text": text, "ocr_quad": quad}]
    return []


def _multi_ocr_core(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float,
    fuzzy_threshold: float,
    *,
    require_quad: bool,
    quad_iou_threshold: float,
) -> Tuple[float, float, float, int, int, int]:
    """Shared association-gated set-F1 loop for both multi-OCR variants.

    For every (pred_player, gt_player) bbox-IoU match, Hungarian-match their
    ``ocr_items``. An item pair is a "hit" iff
    ``levenshtein_similarity(text_pred, text_gt) >= fuzzy_threshold`` and,
    when ``require_quad`` is True, ALSO ``quad_iou >= quad_iou_threshold``.
    Denominators are all GT items (misses hurt recall) and all pred items
    (hallucinations hurt precision). Returns
    ``(precision, recall, f1, total_gt, total_pred, total_hits)``.

    ``require_quad=False`` is the arm-independent variant: E1b
    (``ocr_quads="none"``) emits no quads, so a quad gate would score it 0 for
    the ablated field alone -- a measurement artefact, not a reading-quality
    difference. Texts are already ``normalize_ocr_text``-repaired upstream in
    :func:`_ocr_items_for_player`, so ``.strip().lower()`` here is the visible-
    character comparison the '?'-clean policy intends.
    """
    total_gt = 0
    total_pred = 0
    total_hits = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_players = pred.get("players", []) or []
        gt_players = gt.get("players", []) or []
        if not gt_players:
            continue

        for gp in gt_players:
            total_gt += len(_ocr_items_for_player(gp))
        for pp in pred_players:
            total_pred += len(_ocr_items_for_player(pp))

        if not pred_players:
            continue

        matches = _match_predictions(pred_players, gt_players, iou_threshold)

        for pi, gj in matches.pred_to_gt.items():
            p_items = _ocr_items_for_player(pred_players[pi])
            g_items = _ocr_items_for_player(gt_players[gj])
            if not p_items or not g_items:
                continue

            cost = np.full((len(p_items), len(g_items)), 1e6, dtype=np.float64)
            for i, pi_it in enumerate(p_items):
                p_text = (pi_it.get("text") or "").strip().lower()
                if not p_text:
                    continue
                p_aabb = _quad_pairs_to_aabb(pi_it.get("ocr_quad") or [])
                if require_quad and p_aabb is None:
                    continue
                for j, gj_it in enumerate(g_items):
                    g_text = (gj_it.get("text") or "").strip().lower()
                    if not g_text:
                        continue
                    sim = 1.0 - _normalized_levenshtein(p_text, g_text)
                    if sim < fuzzy_threshold:
                        continue
                    if require_quad:
                        g_aabb = _quad_pairs_to_aabb(gj_it.get("ocr_quad") or [])
                        if g_aabb is None:
                            continue
                        quad_iou = _bbox_iou(p_aabb, g_aabb)
                        if quad_iou < quad_iou_threshold:
                            continue
                        cost[i, j] = -(sim + quad_iou)
                    else:
                        cost[i, j] = -sim

            row_idx, col_idx = linear_sum_assignment(cost)
            for ri, ci in zip(row_idx, col_idx):
                if cost[ri, ci] < 1e5:
                    total_hits += 1

    def _safe(num: int, denom: int) -> float:
        return float(num) / float(denom) if denom > 0 else 0.0

    precision = _safe(total_hits, total_pred)
    recall = _safe(total_hits, total_gt)
    f1 = _safe(2 * precision * recall, max(precision + recall, 1e-9))
    return precision, recall, f1, total_gt, total_pred, total_hits


def multi_ocr_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    fuzzy_threshold: float = OCR_FUZZY_THRESHOLD,
    quad_iou_threshold: float = 0.30,
) -> Dict[str, float]:
    """Quad-GATED set-F1 over the full per-player ``ocr_items`` lists.

    Thin wrapper over :func:`_multi_ocr_core` with ``require_quad=True``: an
    item pair is a hit iff ``quad_iou >= quad_iou_threshold`` AND text similarity
    ``>= fuzzy_threshold``. Captures *dense* OCR coverage in the cycle-3 schema
    (jersey number + name + sponsor logos), complementing ``ocr_metrics`` which
    only scores the primary jersey. Structurally impossible for the E1b no-quad
    arm -- use :func:`multi_ocr_text_metrics` for the cross-arm comparison.
    """
    p, r, f1, g, pr, h = _multi_ocr_core(
        predictions, ground_truths, iou_threshold, fuzzy_threshold,
        require_quad=True, quad_iou_threshold=quad_iou_threshold,
    )
    return {
        "multi_ocr_precision": p,
        "multi_ocr_recall": r,
        "multi_ocr_f1": f1,
        "multi_ocr_gt": g,
        "multi_ocr_pred": pr,
        "multi_ocr_hits": h,
    }


def multi_ocr_text_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    fuzzy_threshold: float = OCR_FUZZY_THRESHOLD,
) -> Dict[str, float]:
    """Quad-INDEPENDENT set F1 over per-player ``ocr_items`` -- the fair E1<->E1b
    metric AND a val_score component.

    Thin wrapper over :func:`_multi_ocr_core` with ``require_quad=False``: a hit
    is text-only (``levenshtein_similarity >= fuzzy_threshold``, no quad gate).
    This is the ONLY OCR metric both arms are compared on, because E1b
    (``ocr_quads="none"``) emits no quads at all; the quad-gated ``multi_ocr_f1``
    would score it 0 purely for the ablated field.
    """
    p, r, f1, g, pr, h = _multi_ocr_core(
        predictions, ground_truths, iou_threshold, fuzzy_threshold,
        require_quad=False, quad_iou_threshold=0.0,
    )
    return {
        "multi_ocr_text_precision": p,
        "multi_ocr_text_recall": r,
        "multi_ocr_text_f1": f1,
        "multi_ocr_text_gt": g,
        "multi_ocr_text_pred": pr,
        "multi_ocr_text_hits": h,
    }


# =========================================================================
# Team clustering (Hungarian matching)
# =========================================================================

def team_clustering_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> Dict[str, float]:
    """Per-frame Hungarian match between predicted clusters and GT teams.

    Avoids the label-permutation problem (predicted "tA" might be GT
    "tB"). For each frame:
      1. IoU-match preds to GTs.
      2. Build a confusion matrix C[pred_team, gt_team].
      3. Find the assignment that maximises the trace via the Hungarian
         algorithm.
    Frame purity = sum(matched cells) / num_matched_players.
    Aggregate is the mean per-frame purity (over frames with >=2 GT teams).
    """
    per_frame_purity: List[float] = []
    total_matched = 0
    total_correct = 0

    label_idx: Dict[str, int] = {}

    def _idx(label: str) -> int:
        if label not in label_idx:
            label_idx[label] = len(label_idx)
        return label_idx[label]

    for pred, gt in zip(predictions, ground_truths):
        pred_players = pred.get("players", []) or []
        gt_players = gt.get("players", []) or []
        if len(gt_players) < 2:
            continue
        matches = _match_predictions(pred_players, gt_players, iou_threshold)
        if not matches.pred_to_gt:
            continue

        labels = sorted({(gp.get("team") or "other") for gp in gt_players})
        if len(labels) < 2:
            continue

        # Confusion matrix [pred_label, gt_label].
        n = max(len(labels), 1)
        confusion = np.zeros((n, n), dtype=np.int32)
        local_pred_idx: Dict[str, int] = {l: i for i, l in enumerate(labels)}
        local_gt_idx = local_pred_idx

        matched_in_frame = 0
        for pi, gj in matches.pred_to_gt.items():
            p_team = (pred_players[pi].get("team") or "other")
            g_team = (gt_players[gj].get("team") or "other")
            pi_idx = local_pred_idx.get(p_team)
            gj_idx = local_gt_idx.get(g_team)
            if pi_idx is None or gj_idx is None:
                # Predicted label is outside GT label set; skip (counts as wrong).
                continue
            confusion[pi_idx, gj_idx] += 1
            matched_in_frame += 1

        if matched_in_frame == 0:
            continue

        # Hungarian: maximise trace -> minimise -confusion.
        row_ind, col_ind = linear_sum_assignment(-confusion)
        correct_in_frame = int(confusion[row_ind, col_ind].sum())
        purity = correct_in_frame / matched_in_frame
        per_frame_purity.append(purity)
        total_matched += matched_in_frame
        total_correct += correct_in_frame

    return {
        "team_purity_mean": float(np.mean(per_frame_purity)) if per_frame_purity else 0.0,
        "team_purity_micro": (total_correct / total_matched) if total_matched else 0.0,
        "team_frames_evaluated": len(per_frame_purity),
        "team_matched_players": total_matched,
    }


# =========================================================================
# Sport-type accuracy (added for the ECCV-2026 cycle-3 schema)
# =========================================================================

def sport_type_accuracy(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Frame-level accuracy of the ``sport_type`` head.

    Compares the case-folded, trimmed string predicted by the model
    (``predictions[i]["sport_type"]``) against the GT label
    (``ground_truths[i]["sport_type"]``). Frames where the prediction is
    an empty string are counted as misses (this lets us tell apart the
    Florence-2 baselines -- which produce empty placeholders -- from the
    sport-aware finetuned variants).

    Returned keys:
        sport_type_accuracy         -- fraction of correct frames (frames
                                       with empty GT excluded from denom).
        sport_type_evaluated_frames -- frames with a non-empty GT.
        sport_type_predicted_frames -- frames where pred is non-empty.
        sport_type_correct          -- exact-match count.
    """
    total = 0
    correct = 0
    predicted = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_label = str(gt.get("sport_type") or "").strip()
        if not gt_label:
            # No GT sport (cycle-2 frames). Skip from accuracy denominator.
            continue
        total += 1
        pred_label = str(pred.get("sport_type") or "").strip()
        if pred_label:
            predicted += 1
        if pred_label.casefold() == gt_label.casefold() and pred_label:
            correct += 1

    return {
        "sport_type_accuracy": (correct / total) if total else 0.0,
        "sport_type_evaluated_frames": total,
        "sport_type_predicted_frames": predicted,
        "sport_type_correct": correct,
    }


# =========================================================================
# Scene-description Sentence-BERT cosine
# =========================================================================

def _load_sbert():
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; scene_sbert_cosine will be None."
        )
        return None
    logger.info(f"Loading Sentence-BERT model: {_SBERT_MODEL_NAME}")
    _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
    return _sbert_model


def scene_description_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Average cosine similarity of scene_description embeddings."""
    pairs = [
        (str(p.get("scene_description") or "").strip(),
         str(g.get("scene_description") or "").strip())
        for p, g in zip(predictions, ground_truths)
    ]
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return {"scene_sbert_cosine_mean": None, "scene_sbert_n": 0}

    model = _load_sbert()
    if model is None:
        return {"scene_sbert_cosine_mean": None, "scene_sbert_n": len(pairs)}

    preds_emb = model.encode([a for a, _ in pairs], convert_to_numpy=True, show_progress_bar=False)
    gts_emb = model.encode([b for _, b in pairs], convert_to_numpy=True, show_progress_bar=False)

    # Cosine via normalize-and-dot.
    preds_emb /= np.linalg.norm(preds_emb, axis=1, keepdims=True) + 1e-12
    gts_emb /= np.linalg.norm(gts_emb, axis=1, keepdims=True) + 1e-12
    cos = (preds_emb * gts_emb).sum(axis=1)

    return {
        "scene_sbert_cosine_mean": float(cos.mean()),
        "scene_sbert_n": int(len(pairs)),
    }


# =========================================================================
# Latency
# =========================================================================

def latency_metrics(
    latencies_ms: Optional[List[float]] = None,
    per_stage_latencies_ms: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Aggregate per-frame wall-clock latency, optionally with stage breakdown."""
    if not latencies_ms:
        return {"latency_ms_mean": None}
    arr = np.asarray([x for x in latencies_ms if x is not None and x > 0], dtype=np.float64)
    if arr.size == 0:
        return {"latency_ms_mean": None}

    out: Dict[str, Any] = {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "latency_n": int(arr.size),
    }
    if per_stage_latencies_ms:
        out["latency_per_stage_ms_mean"] = {
            stage: float(np.mean(vals)) if vals else None
            for stage, vals in per_stage_latencies_ms.items()
        }
    return out


# =========================================================================
# Top-level
# =========================================================================

def compute_all_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    latencies_ms: Optional[List[float]] = None,
    per_stage_latencies_ms: Optional[Dict[str, List[float]]] = None,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Run every metric on (predictions, ground_truths) and return a flat dict.

    The two lists must be aligned: ``predictions[i]`` is the model's
    output for the same frame as ``ground_truths[i]``.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths "
            f"({len(ground_truths)}) lengths differ."
        )

    out: Dict[str, Any] = {
        "num_frames": len(predictions),
        "iou_threshold": iou_threshold,
        "ocr_fuzzy_threshold": OCR_FUZZY_THRESHOLD,
    }
    out.update(average_precision_at_50(predictions, ground_truths, iou_threshold))
    out.update(ocr_metrics(predictions, ground_truths, iou_threshold))
    out.update(multi_ocr_metrics(predictions, ground_truths, iou_threshold))
    # Quad-independent OCR F1 -- the fair E1<->E1b comparison metric (always on).
    out.update(multi_ocr_text_metrics(predictions, ground_truths, iou_threshold))
    out.update(team_clustering_metrics(predictions, ground_truths, iou_threshold))
    out.update(sport_type_accuracy(predictions, ground_truths))
    out.update(scene_description_metrics(predictions, ground_truths))
    out.update(latency_metrics(latencies_ms, per_stage_latencies_ms))
    return out
