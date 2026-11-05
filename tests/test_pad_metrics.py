import math

import pytest

from pad_metrics import apcer, bpcer, bpcer_at_apcer, eer, worst_apcer

# Live scores: higher means bona fide; accepted as bona fide when score >= threshold
BONA_FIDE = [0.9, 0.8, 0.7, 0.4]
ATTACKS = {"print": [0.1, 0.2, 0.6], "replay": [0.3, 0.75]}


def test_apcer_is_the_share_of_attacks_accepted():
    assert apcer(ATTACKS["print"], 0.5) == pytest.approx(1 / 3)
    assert apcer(ATTACKS["print"], 0.6) == pytest.approx(1 / 3)  # equal to the threshold is accepted
    assert apcer(ATTACKS["print"], 0.61) == 0.0
    assert math.isnan(apcer([], 0.5))


def test_bpcer_is_the_share_of_bona_fide_rejected():
    assert bpcer(BONA_FIDE, 0.5) == pytest.approx(1 / 4)
    assert bpcer(BONA_FIDE, 0.95) == 1.0
    assert math.isnan(bpcer([], 0.5))


def test_reported_apcer_is_the_worst_attack_type():
    # ISO/IEC 30107-3 reports the maximum over attack types, not the pooled share
    assert worst_apcer(ATTACKS, 0.5) == (pytest.approx(1 / 2), "replay")
    assert worst_apcer(ATTACKS, 0.65) == (pytest.approx(1 / 2), "replay")
    assert worst_apcer(ATTACKS, 0.8) == (0.0, "print")


def test_eer_finds_the_threshold_where_both_errors_meet():
    value, threshold = eer([0.9, 0.8, 0.6, 0.3], [0.1, 0.2, 0.4, 0.7])
    assert value == pytest.approx(0.25)
    assert apcer([0.1, 0.2, 0.4, 0.7], threshold) <= 0.25
    assert bpcer([0.9, 0.8, 0.6, 0.3], threshold) <= 0.25
    assert eer([0.9, 0.8], [0.1, 0.2]) == (0.0, pytest.approx(0.8))


def test_bpcer_at_apcer_uses_the_worst_attack_type():
    # replay reaches 0.75, so no attack type is accepted only from 0.76 up:
    # BPCER at APCER 0 rejects 0.7 and 0.4
    value, threshold = bpcer_at_apcer(BONA_FIDE, ATTACKS, 0.0)
    assert threshold == pytest.approx(0.8)
    assert value == pytest.approx(2 / 4)
    # APCER up to 50% per type: 0.6 and 0.75 may pass, threshold 0.4 is the lowest
    # where print stays at 1/3 and replay at 1/2
    value, threshold = bpcer_at_apcer(BONA_FIDE, ATTACKS, 0.5)
    assert threshold == pytest.approx(0.4)
    assert value == 0.0
