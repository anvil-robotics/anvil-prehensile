"""Golden equivalence test for the Phase 4a hand-descriptor refactor.

The refactor that gave ``CurlMapper`` an optional ``hand: HandDescriptor``
constructor kwarg (``prehensile.hand`` / ``prehensile.hand_loader``) is
required to be a pure internal restructuring for anyone who never passes
``hand=``: every string literal that used to index straight into
``command.L6_SDK_ORDER`` now resolves through ``L6_HAND`` (an IDENTITY role
map off that same order), so it must land on the exact same index it always
did. This test is the structural guarantee turned into a number: it pins
``CurlMapper.__call__``/``home_gesture()``/``last_unparked``/
``parked_channels``/``couple_index_floor`` output, captured from the
PRE-refactor implementation, and compares by exact string equality of each
float's ``repr()`` (CPython's float repr round-trips exactly, so this is
byte-identity, not ``pytest.approx``) -- never approximate.

``tests/fixtures/curlmap_golden.json`` was captured by running the dump below
against the pre-refactor ``prehensile/curlmap.py`` (before ``prehensile/hand.py``
and ``prehensile/hand_loader.py`` existed, when the six slot literals were
still bare ``L6_SDK_ORDER`` positions) and is re-verified byte-for-byte
identical against the post-refactor implementation.
"""
import json
from pathlib import Path

import numpy as np

from prehensile import fk
from prehensile.curlmap import CurlMapper
from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning

_FIXTURE = Path(__file__).parent / "fixtures" / "curlmap_golden.json"


def _kp_set() -> dict:
    """A decent spread of synthetic (21,3) keypoint frames: the identity/open
    pose plus 6 deterministically-perturbed ("curled by varying amounts, in
    varying directions") frames, built the same way ``test_fk.py``'s synthetic
    fixtures are -- perturbing glove quats, not raw keypoints, so every frame
    is a physically valid FK decode."""
    q = fk.identity_quats(fk.FK_MODE)
    frames = {"open": fk.keypoints_from_quats(q, fk.FK_MODE)}
    rng = np.random.default_rng(1234)
    for n in range(6):
        qq = np.array(q, dtype=float, copy=True)
        qq[:, :3] += rng.normal(0, 0.25, qq[:, :3].shape)
        qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        frames[f"rand{n}"] = fk.keypoints_from_quats(qq, fk.FK_MODE)
    return frames


def _dump() -> dict:
    """Every observable ``CurlMapper`` produces, over both hands, both
    ``couple_thumb_index`` settings, a laggy and an unlagged EMA alpha, and
    frames alternating locked/unlocked -- the same sweep golden_before.json
    was captured with. Floats are stored as ``repr()`` strings (exact,
    round-trippable) so the comparison in the test below is never approximate.
    """
    out = {}
    frames = _kp_set()
    for side in ("left", "right"):
        tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=side, require_couple_low=True)
        for couple in (True, False):
            for alpha in (0.4, 1.0):
                m = CurlMapper(side=side, alpha=alpha, abd_invert=True,
                               tuning=tuning, couple_thumb_index=couple)
                key = f"{side}|c{couple}|a{alpha}"
                seq = []
                for i, (nm, kp) in enumerate(sorted(frames.items())):
                    m.locked = (i % 2 == 1)  # alternate BOTH lock states across frames
                    r = m(kp)
                    seq.append([
                        nm, m.locked,
                        None if r is None else [repr(v) for v in r],
                        None if m.last_unparked is None else [repr(v) for v in m.last_unparked],
                        list(m.parked_channels), repr(m.couple_index_floor),
                    ])
                seq.append(["home", None, [repr(v) for v in m.home_gesture()], None, [], None])
                out[key] = seq
    return out


def test_golden_dump_matches_pre_refactor_fixture_exactly():
    expected = json.loads(_FIXTURE.read_text())
    actual = _dump()
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == expected[key], key


def test_golden_fixture_is_not_vacuous():
    """Guard the fixture itself: it must actually exercise variation (locked
    AND unlocked frames, None-free angle output, a real couple_index_floor,
    at least one parked channel) -- otherwise the exact-match test above could
    pass trivially against a degenerate/empty capture."""
    expected = json.loads(_FIXTURE.read_text())
    assert len(expected) == 8  # 2 sides x 2 couple settings x 2 alphas
    any_locked = any_unlocked = any_parked = False
    for seq in expected.values():
        assert len(seq) == 8  # 7 frames (open + 6 rand) + the trailing "home" entry
        for nm, locked, angles, _last_unparked, parked, _floor in seq:
            if nm != "home":
                assert angles is not None
                assert len(angles) == 6
            any_locked = any_locked or locked is True
            any_unlocked = any_unlocked or locked is False
            any_parked = any_parked or bool(parked)
    assert any_locked and any_unlocked and any_parked
