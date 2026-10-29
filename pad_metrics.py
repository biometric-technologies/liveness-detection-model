"""Presentation attack detection metrics in the terms of ISO/IEC 30107-3.

Scores are "live" scores: higher means more likely a bona fide presentation.
A presentation is accepted as bona fide when score >= threshold.

APCER  share of attack presentations accepted as bona fide (per attack type;
       the reported value is the worst type, as the standard requires)
BPCER  share of bona fide presentations rejected as attacks
ACER   (APCER + BPCER) / 2, common in the literature but not part of the standard
"""
import numpy as np


def apcer(attack_scores, threshold):
    a = np.asarray(attack_scores, dtype=float)
    return float(np.mean(a >= threshold)) if a.size else float('nan')


def bpcer(bona_fide_scores, threshold):
    b = np.asarray(bona_fide_scores, dtype=float)
    return float(np.mean(b < threshold)) if b.size else float('nan')


def worst_apcer(attacks_by_type, threshold):
    """Maximum APCER over attack types and the type that gives it."""
    worst, worst_type = 0.0, None
    for attack_type, scores in attacks_by_type.items():
        value = apcer(scores, threshold)
        if worst_type is None or value > worst:
            worst, worst_type = value, attack_type
    return worst, worst_type


def eer(bona_fide_scores, attack_scores):
    """Equal error rate and its threshold (BPCER = APCER)."""
    b = np.asarray(bona_fide_scores, dtype=float)
    a = np.asarray(attack_scores, dtype=float)
    thresholds = np.unique(np.concatenate([b, a, [np.inf]]))
    best = (1.0, 0.0)
    for t in thresholds:
        e = max(apcer(a, t), bpcer(b, t))
        if e < best[0]:
            best = (e, float(t))
    return best


def bpcer_at_apcer(bona_fide_scores, attacks_by_type, target_apcer):
    """Lowest BPCER whose worst-type APCER does not exceed the target (e.g. BPCER20 for 5%)."""
    all_scores = np.concatenate([np.asarray(bona_fide_scores, dtype=float)] +
                                [np.asarray(s, dtype=float) for s in attacks_by_type.values()] + [[np.inf]])
    for t in np.unique(all_scores):
        if worst_apcer(attacks_by_type, t)[0] <= target_apcer:
            return bpcer(bona_fide_scores, t), float(t)
    return 1.0, float('inf')
