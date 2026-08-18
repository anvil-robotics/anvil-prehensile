"""TDD validators for prehensile/curlmap.py (direct per-finger curl -> L6 map).

SYNTHETIC + fixture-gated, mirroring tests/test_fk.py's conventions: no
conftest, synthetic (21,3) keypoints built via
``fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)`` for the
open hand, and single/all-finger curls built like
``test_fk.test_single_joint_bend_isolation`` (rotate a finger's proximal(_0) +
intermediate(_1) glove quats about a local z axis -- the same construction
``test_fk.test_select_mode_discovery_roundtrip`` uses for its synthetic
"fist"). ``CurlMapper`` is always built from explicit bound dicts, so no test
needs an interactive calibration step -- see ``_fitted_mapper`` below, which
fits those bounds off the synthetic open/fist frames in the TEST (this package
has no ``CurlMapper.calibrate()``; the mapper takes its bounds as constructor
kwargs only).

Note on isolation: because each FK finger chain only consumes its own glove
quat slots, and a node's position depends only on its ancestors' orientations
(never its own joint's), curling one finger leaves every other finger's
landmarks -- and even that finger's OWN base landmark -- bit-identical to rest.
This lets the isolation tests assert exact equality, not just "close".
"""

import math
from pathlib import Path

import numpy as np
import pytest

from prehensile import fk
from prehensile.command import L6_SDK_ORDER
from prehensile.curlmap import (
    _CHORD_FINGERS,
    _FLEX_CHAINS,
    CurlMapper,
    _chain_length,
    _thumb_abd_angle_deg,
    _thumb_flex_bend_deg,
)
from prehensile.profiles import GLOVES

# tuning (yaml) is imported by the TEST only, for the shipped-config guards below;
# curlmap.py itself stays pure-numpy and yaml-free.
from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning

# JointsIndex_Quat proximal(_0) slot per finger (index, middle, ring, pinky, thumb).
_SLOT_BY_FINGER = {"index": 0, "middle": 3, "ring": 6, "pinky": 9, "thumb": 12}


def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    """XYZW quaternion for a rotation of ``angle_rad`` about ``axis`` (any length)."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle_rad / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle_rad / 2.0)])


def _curl_kp(fingers, angle_deg: float = 75.0) -> np.ndarray:
    """Identity-quat keypoints with the given fingers' proximal+intermediate
    joints flexed by ``angle_deg`` about local z (mirrors test_fk's synthetic
    fist construction), decoded to (21,3) under fk.FK_MODE."""
    q = fk.identity_quats(fk.FK_MODE)
    flex = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(angle_deg))
    for finger in fingers:
        base = _SLOT_BY_FINGER[finger]
        q[base] = flex
        q[base + 1] = flex
    return fk.keypoints_from_quats(q, fk.FK_MODE)


def _curl_kp_each(angles_by_finger: dict[str, float]) -> np.ndarray:
    """Like _curl_kp but per-finger, so the three MRP channels reach three
    distinct readings."""
    q = fk.identity_quats(fk.FK_MODE)
    for finger, angle_deg in angles_by_finger.items():
        base = _SLOT_BY_FINGER[finger]
        flex = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(angle_deg))
        q[base] = flex
        q[base + 1] = flex
    return fk.keypoints_from_quats(q, fk.FK_MODE)


_KP_OPEN = fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)
_KP_FIST = _curl_kp(("thumb", "index", "middle", "ring", "pinky"))

_FLEX_SLOTS = ["thumb_flex", "index", "middle", "ring", "pinky"]


def _bounds_from(open_kp: np.ndarray, fist_kp: np.ndarray) -> dict:
    """Per-channel bound kwargs fitted from one open frame and one fist frame.

    The mapper has no ``calibrate()`` method -- bounds are constructor kwargs --
    so the fitting arithmetic lives here in the test instead: each chord
    finger's r_open/r_closed from the two frames' chord ratios, and the thumb's
    flex/abduction bounds from its two dedicated metrics. The four thumb poses
    are taken from the same open/fist frames (extended & abduct <- OPEN, curl &
    adduct <- FIST): only relative separation matters for the arithmetic
    verified here; the live isolated gestures provide the true anatomical
    mapping (see curlmap.py's module docstring).
    """
    # float64 like __call__'s own `np.asarray(kp, dtype=np.float64)`: fk hands
    # back float32, and fitting in float32 leaves the endpoints a few 1e-6 off
    # (enough to fail the exact-endpoint assertions below).
    open_kp = np.asarray(open_kp, dtype=np.float64)
    fist_kp = np.asarray(fist_kp, dtype=np.float64)
    r_open: dict[str, float] = {}
    r_closed: dict[str, float] = {}
    for f in _CHORD_FINGERS:
        chain = _FLEX_CHAINS[f]
        r_open[f] = float(
            np.linalg.norm(open_kp[chain[-1]] - open_kp[chain[0]])
        ) / _chain_length(open_kp, chain)
        r_closed[f] = float(
            np.linalg.norm(fist_kp[chain[-1]] - fist_kp[chain[0]])
        ) / _chain_length(fist_kp, chain)
    return {
        "r_open": r_open,
        "r_closed": r_closed,
        "thumb_flex_bounds": {
            "open": _thumb_flex_bend_deg(open_kp),
            "closed": _thumb_flex_bend_deg(fist_kp),
        },
        "abd_bounds": {
            "spread": _thumb_abd_angle_deg(open_kp),
            "tuck": _thumb_abd_angle_deg(fist_kp),
        },
    }


_SYNTH_BOUNDS = _bounds_from(_KP_OPEN, _KP_FIST)


def _fitted_mapper(alpha: float = 1.0, **kwargs) -> CurlMapper:
    """A CurlMapper whose bounds are fitted on the synthetic open/fist frames
    (see ``_bounds_from``), so _KP_OPEN reads ~100 and _KP_FIST reads ~0 on
    every flex channel. Extra ``kwargs`` pass through to CurlMapper (e.g.
    flex_gain / abd_gain / tuning)."""
    return CurlMapper(alpha=alpha, **_SYNTH_BOUNDS, **kwargs)


def _hand_with_thumb_abduction(abd_deg: float) -> np.ndarray:
    """Flat-fingered hand whose thumb sits at a known palmar abduction, so
    ``_thumb_abd_angle_deg`` reads back exactly ``abd_deg``."""
    kp = np.zeros((21, 3), dtype=np.float64)
    for base, x in [(5, -0.03), (9, -0.01), (13, 0.01), (17, 0.03)]:
        for k in range(4):
            kp[base + k] = [x, 0.03 * (k + 1), 0.0]
    # Palm normal is -z with the fingers flat, so +abd_deg is -sin(a) in z.
    a = math.radians(abd_deg)
    direction = np.array([0.0, math.cos(a), -math.sin(a)])
    kp[1] = np.array([-0.04, 0.0, 0.0])
    for k in range(3):
        kp[k + 2] = kp[k + 1] + 0.025 * direction
    return kp


# -- core mapping: shape, range, monotonicity, isolation ----------------------- #


def test_open_hand_all_flex_channels_high():
    mapper = _fitted_mapper()
    angles = mapper(_KP_OPEN)
    assert angles is not None
    by_slot = dict(zip(L6_SDK_ORDER, angles))
    for slot in _FLEX_SLOTS:
        assert by_slot[slot] == pytest.approx(100.0, abs=1e-6), f"{slot}={by_slot[slot]}"


def test_curl_one_finger_isolated():
    """Curling only the index finger drops only the index channel; the other
    four flex channels and thumb_abd stay at their open (~100) values."""
    mapper = _fitted_mapper()
    kp_index_curled = _curl_kp(("index",))
    angles = mapper(kp_index_curled)
    assert angles is not None
    by_slot = dict(zip(L6_SDK_ORDER, angles))
    assert by_slot["index"] == pytest.approx(0.0, abs=1e-6)
    for slot in ("thumb_flex", "middle", "ring", "pinky", "thumb_abd"):
        assert by_slot[slot] == pytest.approx(100.0, abs=1e-6), f"{slot}={by_slot[slot]}"


def test_monotonic_flexion_sweep():
    """Sweeping one finger's flexion 0->90 deg never increases its channel.

    (Verified separately that this chain's chord ratio is non-monotonic PAST
    ~105 deg -- it folds back on itself -- so the sweep stays inside the
    confirmed-monotonic 0-90 deg range.)
    """
    mapper = CurlMapper(alpha=1.0)  # alpha=1 => no EMA lag between samples
    values = []
    for angle_deg in (0, 15, 30, 45, 60, 75, 90):
        kp = _curl_kp(("index",), angle_deg=angle_deg)
        angles = mapper(kp)
        assert angles is not None
        values.append(dict(zip(L6_SDK_ORDER, angles))["index"])
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:])), values


def test_output_shape_range_and_sdk_order():
    mapper = _fitted_mapper()
    angles = mapper(_KP_OPEN)
    assert angles is not None
    assert len(angles) == 6
    assert all(0.0 <= a <= 100.0 for a in angles)
    assert L6_SDK_ORDER == ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]


def test_fitted_bounds_map_open_high_fist_low():
    """The fitted bounds put the two frames they were fitted on at the two
    endpoints: open -> 100, fist -> 0 on every flex channel. (This was
    ``test_calibrate_fits_open_high_fist_low`` where a ``calibrate()`` method
    existed; here the same arithmetic is exercised through the constructor's
    bound kwargs.)"""
    open_angles = dict(zip(L6_SDK_ORDER, _fitted_mapper()(_KP_OPEN)))
    fist_angles = dict(zip(L6_SDK_ORDER, _fitted_mapper()(_KP_FIST)))  # fresh EMA state
    for slot in _FLEX_SLOTS:
        assert open_angles[slot] == pytest.approx(100.0, abs=1e-6)
        assert fist_angles[slot] == pytest.approx(0.0, abs=1e-6)


# -- flips, inverts and the response knobs ------------------------------------ #


def test_flip_tuning_complements_channel():
    """A ``tuning`` ``flip: true`` complements that channel's output (100-x),
    leaving every other channel identical (the mirrored-thumb knob)."""
    kp = _curl_kp(("index",), angle_deg=45)
    base = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))
    flipped = dict(zip(L6_SDK_ORDER, CurlMapper(
        alpha=1.0, tuning={"thumb_flex": {"flip": True}, "thumb_abd": {"flip": True}}
    )(kp)))
    assert flipped["thumb_flex"] == pytest.approx(100.0 - base["thumb_flex"], abs=1e-6)
    assert flipped["thumb_abd"] == pytest.approx(100.0 - base["thumb_abd"], abs=1e-6)
    for slot in ("index", "middle", "ring", "pinky"):
        assert flipped[slot] == pytest.approx(base[slot], abs=1e-9)


def test_side_alone_does_not_flip():
    """``side`` no longer auto-flips the thumb; the flip must come from tuning."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    assert CurlMapper(alpha=1.0, side="right")(kp) == pytest.approx(
        CurlMapper(alpha=1.0, side="left")(kp)
    )


def test_abd_invert_flips_thumb_abd_only():
    """abd_invert flips thumb_abd to the complementary value and leaves every
    flex channel untouched (the Wuji reversed-abduction fix)."""
    bounds = {"spread": 55.0, "tuck": 35.0}
    base = CurlMapper(alpha=1.0, abd_bounds=bounds)
    inv = CurlMapper(alpha=1.0, abd_bounds=bounds, abd_invert=True)
    a = dict(zip(L6_SDK_ORDER, base(_KP_FIST)))
    b = dict(zip(L6_SDK_ORDER, inv(_KP_FIST)))
    assert b["thumb_abd"] == pytest.approx(100.0 - a["thumb_abd"], abs=1e-6)
    for slot in ("thumb_flex", "index", "middle", "ring", "pinky"):
        assert b[slot] == pytest.approx(a[slot], abs=1e-9)


def test_abd_invert_complements_only_the_thumb_abd_channel():
    """``abd_invert`` complements thumb_abd and leaves every other channel alone."""
    kp = _hand_with_thumb_abduction(20.0)   # ~71% under ABD_TUCK/ABD_SPREAD
    i = L6_SDK_ORDER.index("thumb_abd")

    plain = CurlMapper(side="right", abd_invert=False)(kp)
    inverted = CurlMapper(side="right", abd_invert=True)(kp)

    # Guard the fixture: a complement at a bound or at the midpoint is invisible.
    assert 0.0 < plain[i] < 100.0
    assert abs(plain[i] - 50.0) > 1.0

    assert inverted[i] == pytest.approx(100.0 - plain[i], abs=1e-9)
    for j, slot in enumerate(L6_SDK_ORDER):
        if j != i:
            assert inverted[j] == pytest.approx(plain[j], abs=1e-9), slot


def test_wuji_thumb_abd_commands_the_opposite_of_udcap():
    """The two gloves need opposite thumb_abd senses, so one abducted-thumb
    frame commanding N on UDCap must command 100-N on Wuji."""
    kp = _hand_with_thumb_abduction(20.0)
    i = L6_SDK_ORDER.index("thumb_abd")

    def command(glove: str) -> float:
        return CurlMapper(side="right", abd_invert=GLOVES[glove].abd_invert)(kp)[i]

    udcap = command("udcap")
    assert 0.0 < udcap < 100.0            # guard: saturation would hide a flip
    assert abs(udcap - 50.0) > 1.0        # guard: so would the midpoint
    assert command("wuji") == pytest.approx(100.0 - udcap, abs=1e-9)


def test_flex_gain_amplifies_swing():
    """flex_gain>1 pushes a mid-range flex channel further from the 50 midpoint."""
    kp = _curl_kp(("index",), angle_deg=60)  # partial curl -> mid-range (~29)
    base = dict(zip(L6_SDK_ORDER, _fitted_mapper()(kp)))["index"]
    amp = dict(zip(L6_SDK_ORDER, _fitted_mapper(flex_gain=2.0)(kp)))["index"]
    assert base != pytest.approx(50.0, abs=1.0)          # frame must exercise the channel
    assert abs(amp - 50.0) > abs(base - 50.0)


def test_thumb_flex_pivot_pushes_low_end_lower():
    """Raising the thumb_flex gain pivot pushes sub-pivot (curled) values lower,
    and touches only thumb_flex (not the four fingers)."""
    kp = _curl_kp(("thumb",), angle_deg=45)  # partial thumb curl -> mid-low thumb_flex
    base = dict(zip(L6_SDK_ORDER, _fitted_mapper(flex_gain=2.0)(kp)))
    hi = dict(zip(L6_SDK_ORDER, _fitted_mapper(flex_gain=2.0, thumb_flex_pivot=70.0)(kp)))
    assert base["thumb_flex"] < 50.0            # this frame sits below the default pivot
    assert hi["thumb_flex"] < base["thumb_flex"]  # higher pivot pushes it lower still
    for slot in ("index", "middle", "ring", "pinky"):
        assert hi[slot] == pytest.approx(base[slot], abs=1e-9)


def test_tuning_overrides_gain_per_channel():
    """A ``tuning`` dict overrides gain/pivot for the named channel only."""
    kp = _curl_kp(("index",), angle_deg=60)  # mid-range index (~29)
    base = dict(zip(L6_SDK_ORDER, _fitted_mapper()(kp)))
    tuned = dict(zip(L6_SDK_ORDER, _fitted_mapper(tuning={"index": {"gain": 2.0, "pivot": 50.0}})(kp)))
    assert base["index"] != pytest.approx(50.0, abs=1.0)
    assert abs(tuned["index"] - 50.0) > abs(base["index"] - 50.0)   # index amplified
    for slot in ("thumb_flex", "middle", "ring", "pinky", "thumb_abd"):
        assert tuned[slot] == pytest.approx(base[slot], abs=1e-9)   # others untouched


def test_abd_gain_softens_swing():
    """abd_gain<1 pulls thumb_abd toward the 50 midpoint (gentler abduction)."""
    base = dict(zip(L6_SDK_ORDER, _fitted_mapper()(_KP_FIST)))["thumb_abd"]
    soft = dict(zip(L6_SDK_ORDER, _fitted_mapper(abd_gain=0.5)(_KP_FIST)))["thumb_abd"]
    assert base != pytest.approx(50.0, abs=1.0)
    assert abs(soft - 50.0) < abs(base - 50.0)
    # the flex channels are untouched by abd_gain
    base_by = dict(zip(L6_SDK_ORDER, _fitted_mapper()(_KP_FIST)))
    soft_by = dict(zip(L6_SDK_ORDER, _fitted_mapper(abd_gain=0.5)(_KP_FIST)))
    for slot in ("thumb_flex", "index", "middle", "ring", "pinky"):
        assert soft_by[slot] == pytest.approx(base_by[slot], abs=1e-9)


def test_per_channel_alpha_overrides_the_fallback_for_that_slot_only():
    """A tuning "alpha" on one channel reaches that slot's EMA alpha (see
    CurlMapper.__init__'s ``self._alphas``); every other slot keeps the
    scalar ``alpha`` fallback passed to the constructor."""
    mapper = CurlMapper(side="left", alpha=0.4, tuning={"thumb_abd": {"alpha": 0.9}})
    i_thumb_abd = L6_SDK_ORDER.index("thumb_abd")
    for i, slot in enumerate(L6_SDK_ORDER):
        if i == i_thumb_abd:
            assert mapper._alphas[i] == 0.9
        else:
            assert mapper._alphas[i] == 0.4


def test_per_channel_alpha_smooths_thumb_abd_only():
    """A lower per-channel ``alpha`` on thumb_abd slows that channel's response
    without touching the flex channels. (The de-jitter knob is a tuning key in
    this package, not an ``abd_alpha`` constructor kwarg.)"""
    default = _fitted_mapper(alpha=0.5)
    smoothed = _fitted_mapper(alpha=0.5, tuning={"thumb_abd": {"alpha": 0.1}})
    seed_d = dict(zip(L6_SDK_ORDER, default(_KP_OPEN)))
    seed_s = dict(zip(L6_SDK_ORDER, smoothed(_KP_OPEN)))
    step_d = dict(zip(L6_SDK_ORDER, default(_KP_FIST)))
    step_s = dict(zip(L6_SDK_ORDER, smoothed(_KP_FIST)))
    # thumb_abd moves less under the lower alpha...
    assert abs(step_s["thumb_abd"] - seed_s["thumb_abd"]) < abs(step_d["thumb_abd"] - seed_d["thumb_abd"])
    # ...while a flex channel steps identically (same global alpha).
    assert step_s["index"] == pytest.approx(step_d["index"], abs=1e-9)


# -- invalid input + EMA state ------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_kp",
    [
        np.zeros((20, 3)),           # wrong shape
        np.full((21, 3), np.nan),    # non-finite
        np.zeros((21, 3)),           # all L_f == 0 (degenerate)
    ],
)
def test_none_on_invalid_input_leaves_last_unchanged(bad_kp):
    mapper = _fitted_mapper()
    first = mapper(_KP_OPEN)
    assert mapper(bad_kp) is None
    assert mapper._last == first  # EMA state untouched by the failed call
    second = mapper(_KP_OPEN)  # resumes from the untouched state
    assert second == first


@pytest.mark.parametrize(
    "bad_kp",
    [
        np.zeros((20, 3)),           # wrong shape
        np.full((21, 3), np.nan),    # non-finite
        np.zeros((21, 3)),           # all L_f == 0 (degenerate)
    ],
)
def test_last_unparked_unchanged_on_invalid_input(bad_kp):
    """Mirrors test_none_on_invalid_input_leaves_last_unchanged: the early
    ``return None`` paths (bad shape/non-finite/degenerate chain) happen before
    the clip+flip stage, so they must not touch last_unparked either."""
    mapper = _fitted_mapper()
    first = mapper(_KP_OPEN)
    assert first is not None
    first_last_unparked = mapper.last_unparked
    assert mapper(bad_kp) is None
    assert mapper.last_unparked == first_last_unparked
    second = mapper(_KP_OPEN)  # resumes from the untouched state
    assert second == first
    assert mapper.last_unparked == pytest.approx(first_last_unparked, abs=1e-9)


def test_ema_first_frame_unlagged_and_constant_converges():
    mapper = _fitted_mapper(alpha=0.4)
    first = mapper(_KP_OPEN)
    assert first == pytest.approx([100.0] * 6, abs=1e-6)  # unlagged first frame
    again = mapper(_KP_OPEN)
    assert again == pytest.approx(first, abs=1e-9)  # constant input: no drift


def test_ema_alternating_frames_settle_between_extremes():
    mapper = _fitted_mapper(alpha=0.4)
    mapper(_KP_OPEN)
    result = mapper(_KP_FIST)
    by_slot = dict(zip(L6_SDK_ORDER, result))
    assert 0.0 < by_slot["index"] < 100.0


# -- park lock ----------------------------------------------------------------- #


def test_park_forces_value_when_locked():
    """A locked park overrides thumb_flex only, bypassing gain/pivot/EMA lag.

    ``couple_thumb_index=False``: the thumb<-index pinch is ON by default in
    this package and is applied AFTER the park loop, so it would otherwise
    overwrite exactly the channel under test (that precedence is its own test,
    ``test_couple_wins_over_a_park_on_thumb_flex``)."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}},
                   couple_thumb_index=False)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    assert by_slot["thumb_flex"] == pytest.approx(15.0, abs=1e-6)
    unlocked = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))
    for slot in ("index", "middle", "ring", "pinky"):
        assert by_slot[slot] == pytest.approx(unlocked[slot], abs=1e-9)


def test_park_ignored_when_unlocked():
    """The same park tuning has no effect while locked is False."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}})
    m.locked = False
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    no_park = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))
    assert by_slot["thumb_flex"] == pytest.approx(no_park["thumb_flex"], abs=1e-9)


def test_park_is_literal_output_after_flip():
    """park is a fixed OUTPUT target applied AFTER the flip: a flipped channel
    still parks to the literal value (not its 100-x complement), so both hands
    are commanded the same value for the same park."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0, "flip": True}},
                   couple_thumb_index=False)   # see test_park_forces_value_when_locked
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    assert by_slot["thumb_flex"] == pytest.approx(15.0, abs=1e-6)


def test_park_only_affects_channels_with_park():
    """Locking with a park set only on thumb_flex leaves thumb_abd unaffected."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}})
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    unlocked = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))
    assert by_slot["thumb_abd"] == pytest.approx(unlocked["thumb_abd"], abs=1e-9)


def test_parked_channels_empty_when_none_configured():
    """No tuning ``park`` entries and no ``set_park`` calls -> empty tuple."""
    assert CurlMapper(alpha=1.0).parked_channels == ()


def test_parked_channels_reflects_configured_tuning_in_sdk_order():
    """parked_channels lists the tuning-configured park slots in L6_SDK_ORDER
    order (thumb_flex, thumb_abd, index, middle, ring, pinky), independent of
    the order they happen to appear in the tuning dict -- here "pinky" is
    listed first in the dict but must still come out last."""
    m = CurlMapper(alpha=1.0, tuning={"pinky": {"park": 10.0}, "thumb_flex": {"park": 5.0}})
    assert m.parked_channels == ("thumb_flex", "pinky")


def test_parked_channels_updates_live_via_set_park():
    """set_park() adding a park is immediately visible in parked_channels,
    again in L6_SDK_ORDER order rather than call order (thumb_abd is set
    second here but precedes middle in L6_SDK_ORDER)."""
    m = CurlMapper(alpha=1.0)
    assert m.parked_channels == ()
    m.set_park("middle", 20.0)
    assert m.parked_channels == ("middle",)
    m.set_park("thumb_abd", 5.0)
    assert m.parked_channels == ("thumb_abd", "middle")


def test_parked_channels_drops_slot_when_cleared_via_set_park_none():
    """set_park(slot, None) clears that slot out of parked_channels, leaving
    any other configured parks untouched."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}, "pinky": {"park": 1.0}})
    assert m.parked_channels == ("thumb_flex", "pinky")
    m.set_park("thumb_flex", None)
    assert m.parked_channels == ("pinky",)


def test_set_park_updates_live():
    """set_park() can seed/change a channel's parked value at runtime."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, couple_thumb_index=False)  # see test_park_forces_value_when_locked
    m.locked = True
    m.set_park("thumb_flex", 30.0)
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    assert by_slot["thumb_flex"] == pytest.approx(30.0, abs=1e-6)


def test_last_unparked_equals_output_when_unlocked():
    """While ``locked`` is False the park block never runs, so ``last_unparked``
    must mirror the returned output exactly, even when a park value IS
    configured (it just isn't being applied)."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}})
    angles = m(kp)
    assert angles is not None
    assert m.last_unparked == pytest.approx(angles, abs=1e-9)


def test_last_unparked_diverges_from_output_on_parked_slot_when_locked():
    """While locked, the parked slot's ``last_unparked`` entry keeps the
    pre-park TRACKED value (matching what an unparked mapper would have
    produced), even though the returned output is forced to the park value.
    Every other slot's ``last_unparked`` still matches the output exactly."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}},
                   couple_thumb_index=False)   # see test_park_forces_value_when_locked
    m.locked = True
    angles = m(kp)
    assert angles is not None
    by_slot = dict(zip(L6_SDK_ORDER, angles))
    tracked = dict(zip(L6_SDK_ORDER, m.last_unparked))

    # the unparked reference: identical construction/input, just never locked.
    reference = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))

    assert by_slot["thumb_flex"] == pytest.approx(15.0, abs=1e-6)  # forced park
    assert tracked["thumb_flex"] == pytest.approx(reference["thumb_flex"], abs=1e-6)
    assert tracked["thumb_flex"] != pytest.approx(15.0, abs=1.0)  # actually diverges
    for slot in ("thumb_abd", "index", "middle", "ring", "pinky"):
        assert tracked[slot] == pytest.approx(by_slot[slot], abs=1e-9)


def test_last_unparked_is_independent_copy_not_aliased():
    """``last_unparked`` must be a distinct list object from the returned
    output, not an alias of it -- the park block mutates the returned list IN
    PLACE, so aliasing would let that mutation silently corrupt last_unparked
    too. Pin this down directly (object identity + mutate-and-check), rather
    than relying on the numeric divergence test to catch it incidentally."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}})
    m.locked = True
    result = m(kp)
    assert result is not m.last_unparked
    tracked_thumb_flex = m.last_unparked[0]  # thumb_flex is L6_SDK_ORDER[0]
    result[0] = -1234.0  # mutate the returned list directly, as a caller might
    assert m.last_unparked[0] == pytest.approx(tracked_thumb_flex, abs=1e-9)


# --------------------------------------------------------------------------- #
# DEFERRED real-glove regression guard, mirroring test_fk.py's
# test_live_fixture_discovery gating. Offline verification already confirmed
# both fixtures separate open>=80 / fist<=40 on every flex channel using the
# module's fallback R_OPEN/R_CLOSED (four fingers) +
# THUMB_FLEX_OPEN/CLOSED_DEG (thumb) constants.
# --------------------------------------------------------------------------- #
_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_OPEN = _FIXTURES / "quat_open_3s.bin"
_FIXTURE_FIST = _FIXTURES / "quat_fist_3s.bin"
_HAVE_FIXTURES = _FIXTURE_OPEN.exists() and _FIXTURE_FIST.exists()


@pytest.mark.skipif(
    not _HAVE_FIXTURES,
    reason="real glove recordings (tests/fixtures/quat_*_3s.bin) not available",
)
def test_live_fixture_curl_channels_separate():
    from prehensile.udcap import _extract_quats, iter_datagrams

    def frames(path):
        out = []
        for d in iter_datagrams(path):
            result = _extract_quats(d, "left")
            if result is not None:
                out.append(result[0])  # (quats, aButton, bButton) -> quats
        return out

    open_quats = frames(_FIXTURE_OPEN)
    fist_quats = frames(_FIXTURE_FIST)
    assert len(open_quats) > 100 and len(fist_quats) > 100

    mean_open_kp = fk.keypoints_from_quats(np.mean(open_quats, axis=0), fk.FK_MODE)
    mean_fist_kp = fk.keypoints_from_quats(np.mean(fist_quats, axis=0), fk.FK_MODE)

    mapper_open = CurlMapper(alpha=1.0)  # module fallback R_OPEN/R_CLOSED
    open_angles = dict(zip(L6_SDK_ORDER, mapper_open(mean_open_kp)))
    mapper_fist = CurlMapper(alpha=1.0)
    fist_angles = dict(zip(L6_SDK_ORDER, mapper_fist(mean_fist_kp)))
    for slot in _FLEX_SLOTS:
        assert open_angles[slot] >= 80.0, f"{slot} open={open_angles[slot]}"
        assert fist_angles[slot] <= 40.0, f"{slot} fist={fist_angles[slot]}"


# -- thumb<-index coupling (couple_thumb_index + couple_low) --------------------- #

# A half-curled index gives an index reading strictly between 0 and 100, so the
# rescale below is a real interior point rather than a degenerate endpoint.
_KP_INDEX_HALF = _curl_kp(("index",), angle_deg=45)


@pytest.mark.parametrize("low", [0.0, 30.0, 50.0])
def test_couple_maps_index_linearly_onto_couple_low_range(low):
    """While coupled, thumb_flex == low + (index/100) * (100 - low), read off the
    FINAL commanded index. Pins the exact formula at several bounds."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": low}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    expected = low + (by_slot["index"] / 100.0) * (100.0 - low)
    assert by_slot["thumb_flex"] == pytest.approx(expected, abs=1e-6)


def test_couple_low_defaults_to_zero_when_unset():
    """With no couple_low configured the bound is 0, so thumb_flex mirrors index exactly."""
    m = CurlMapper(alpha=1.0, couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    assert by_slot["thumb_flex"] == pytest.approx(by_slot["index"], abs=1e-6)


@pytest.mark.parametrize(("locked", "enabled"), [(False, False), (False, True), (True, False)])
def test_couple_inert_unless_both_locked_and_enabled(locked, enabled):
    """Coupling needs BOTH the runtime lock and the opt-in flag; any other
    combination leaves thumb_flex on its own tracked metric."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                   couple_thumb_index=enabled)
    m.locked = locked
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    plain = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_INDEX_HALF)))
    assert by_slot["thumb_flex"] == pytest.approx(plain["thumb_flex"], abs=1e-9)


def test_couple_wins_over_a_park_on_thumb_flex():
    """Coupling is applied AFTER the park override, so the narrower opt-in wins
    when a channel somehow carries both."""
    m = CurlMapper(alpha=1.0,
                   tuning={"thumb_flex": {"park": 15.0, "couple_low": 30.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    expected = 30.0 + (by_slot["index"] / 100.0) * 70.0
    assert by_slot["thumb_flex"] == pytest.approx(expected, abs=1e-6)
    assert by_slot["thumb_flex"] != pytest.approx(15.0, abs=1e-6)


def test_couple_leaves_last_unparked_tracking_the_real_thumb():
    """last_unparked snapshots BEFORE the coupling, so a UI can still show what
    the operator's actual thumb is doing while the hand is driven by the index."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                   couple_thumb_index=True)
    m.locked = True
    out = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    tracked = dict(zip(L6_SDK_ORDER, m.last_unparked))
    plain = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_INDEX_HALF)))
    assert tracked["thumb_flex"] == pytest.approx(plain["thumb_flex"], abs=1e-9)
    assert out["thumb_flex"] != pytest.approx(tracked["thumb_flex"], abs=1e-6)


def test_couple_does_not_corrupt_ema_so_unlock_has_no_jump():
    """The coupled write lands post-EMA, so thumb_flex's filter keeps tracking the
    real thumb underneath: unlocking resumes exactly where a never-locked mapper
    would be, with no jump."""
    tuning = {"thumb_flex": {"couple_low": 30.0}}
    never = CurlMapper(alpha=0.4, tuning=tuning, couple_thumb_index=True)
    toggled = CurlMapper(alpha=0.4, tuning=tuning, couple_thumb_index=True)
    toggled.locked = True
    for _ in range(3):  # same frames through both; only `toggled` is locked
        never(_KP_INDEX_HALF)
        toggled(_KP_INDEX_HALF)
    toggled.locked = False
    assert dict(zip(L6_SDK_ORDER, toggled(_KP_INDEX_HALF)))["thumb_flex"] == pytest.approx(
        dict(zip(L6_SDK_ORDER, never(_KP_INDEX_HALF)))["thumb_flex"], abs=1e-9)


def test_couple_with_no_tuning_at_all_defaults_low_to_zero():
    """resolve_tuning returns None when there is nothing to apply, so the mapper
    must accept tuning=None without crashing and fall back to a 0 bound."""
    m = CurlMapper(alpha=1.0, tuning=None, couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    assert by_slot["thumb_flex"] == pytest.approx(by_slot["index"], abs=1e-6)


def test_couple_open_index_gives_fully_open_thumb():
    """Endpoint: a fully-open index (100) maps to a fully-open thumb (100),
    independent of the bound."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_OPEN)))
    assert by_slot["index"] == pytest.approx(100.0, abs=1e-6)
    assert by_slot["thumb_flex"] == pytest.approx(100.0, abs=1e-6)


def test_couple_applies_thumb_flex_flip():
    """The coupled value is a TRACKED quantity in physical-openness space, NOT a
    literal SDK command like park -- so it must go through thumb_flex's flip.
    Otherwise a thumb whose mounting is mirrored (flip: true) would close as the
    index opens. This is the one behaviour that differs from park's."""
    plain = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                       couple_thumb_index=True)
    flipped = CurlMapper(alpha=1.0,
                         tuning={"thumb_flex": {"couple_low": 30.0, "flip": True}},
                         couple_thumb_index=True)
    plain.locked = flipped.locked = True
    v_plain = dict(zip(L6_SDK_ORDER, plain(_KP_INDEX_HALF)))["thumb_flex"]
    v_flipped = dict(zip(L6_SDK_ORDER, flipped(_KP_INDEX_HALF)))["thumb_flex"]
    assert v_flipped == pytest.approx(100.0 - v_plain, abs=1e-6)


def test_couple_reads_index_before_the_index_flip():
    """The index is sampled in physical-openness space (pre-flip), so flipping the
    index channel's own output does not invert what the thumb follows."""
    plain = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                       couple_thumb_index=True)
    idx_flipped = CurlMapper(
        alpha=1.0,
        tuning={"thumb_flex": {"couple_low": 30.0}, "index": {"flip": True}},
        couple_thumb_index=True)
    plain.locked = idx_flipped.locked = True
    v_plain = dict(zip(L6_SDK_ORDER, plain(_KP_INDEX_HALF)))["thumb_flex"]
    v_idx_flipped = dict(zip(L6_SDK_ORDER, idx_flipped(_KP_INDEX_HALF)))["thumb_flex"]
    assert v_idx_flipped == pytest.approx(v_plain, abs=1e-6)


# A heavy index curl drives the index channel to 0.0 under default gain/pivot,
# well below any floor below, so the clamp is unambiguously exercised.
_KP_INDEX_SHUT = _curl_kp(("index",), angle_deg=80)
_COUPLE_BOTH = {"thumb_flex": {"couple_low": 15.0}, "index": {"couple_low": 20.0}}


def test_couple_index_floor_clamps_the_commanded_index():
    """couple_low on `index` floors what the index finger itself is commanded, so
    it stops short instead of closing all the way."""
    m = CurlMapper(alpha=1.0, tuning=_COUPLE_BOTH, couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert dict(zip(L6_SDK_ORDER, m.last_unparked))["index"] == pytest.approx(0.0, abs=1e-6)
    assert by_slot["index"] == pytest.approx(20.0, abs=1e-6)


def test_couple_thumb_spends_full_range_over_the_index_working_window():
    """The index floor narrows the index's WORKING window to [index_low, 100], and
    the thumb is normalized within that window -- so it still spends its whole
    [thumb_low, 100] range and thumb_low stays the true floor. Both saturate
    together at the bottom, but no thumb travel is lost to the clamp."""
    m = CurlMapper(alpha=1.0, tuning=_COUPLE_BOTH, couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert by_slot["index"] == pytest.approx(20.0, abs=1e-6)   # index at its floor
    assert by_slot["thumb_flex"] == pytest.approx(15.0, abs=1e-6)  # thumb at ITS floor


def test_couple_thumb_low_is_the_true_floor_whatever_the_index_floor():
    """Whatever the index floor, a fully-shut index bottoms the thumb at exactly
    thumb_low -- the index floor must NOT silently raise it."""
    for index_low in (0.0, 20.0, 50.0, 80.0):
        m = CurlMapper(alpha=1.0,
                       tuning={"thumb_flex": {"couple_low": 30.0},
                               "index": {"couple_low": index_low}},
                       couple_thumb_index=True)
        m.locked = True
        thumb = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))["thumb_flex"]
        assert thumb == pytest.approx(30.0, abs=1e-6), f"index_low={index_low}"


def test_couple_window_remap_is_linear_across_the_window():
    """Mid-window the thumb is a straight line in the normalized index:
    thumb == thumb_low + (idx - index_low)/(100 - index_low) * (100 - thumb_low)."""
    kp = _curl_kp(("index",), angle_deg=35)
    m = CurlMapper(alpha=1.0,
                   tuning={"thumb_flex": {"couple_low": 30.0}, "index": {"couple_low": 20.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))
    idx = by_slot["index"]
    assert 20.0 < idx < 100.0, f"need an interior index for this test, got {idx}"
    expected = 30.0 + ((idx - 20.0) / 80.0) * 70.0
    assert by_slot["thumb_flex"] == pytest.approx(expected, abs=1e-6)


def test_couple_index_floor_of_zero_matches_no_index_floor():
    """An explicit index floor of 0 is the same as omitting it (no clamp, full window)."""
    a = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 30.0}},
                   couple_thumb_index=True)
    b = CurlMapper(alpha=1.0,
                   tuning={"thumb_flex": {"couple_low": 30.0}, "index": {"couple_low": 0.0}},
                   couple_thumb_index=True)
    a.locked = b.locked = True
    kp = _curl_kp(("index",), angle_deg=35)
    assert a(kp) == pytest.approx(b(kp), abs=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    ("flip_thumb", "flip_index"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_couple_flip_matrix_is_physically_identical_on_both_hands(side, flip_thumb, flip_index):
    """BOTH hands x every thumb_flex/index flip combination must put the thumb in
    the SAME PHYSICAL pose -- a flip changes only the SDK encoding of that pose,
    never the pose itself.

    This is the invariant that actually matters on hardware and the one with teeth
    today: the right thumb is already mounted mirrored, so `thumb_flex: {flip:
    true}` is a live possibility, and getting it wrong means the thumb closing as
    the index opens. Decoding the output back to physical openness and comparing
    against an unflipped reference mapper catches both halves of the rule --
    reading the index pre-flip, and applying thumb_flex's flip on write.
    """
    bounds = {"thumb_flex": {"couple_low": 30.0}, "index": {"couple_low": 20.0}}
    tuning = {
        "thumb_flex": {**bounds["thumb_flex"], **({"flip": True} if flip_thumb else {})},
        "index": {**bounds["index"], **({"flip": True} if flip_index else {})},
    }
    m = CurlMapper(side=side, alpha=1.0, tuning=tuning, couple_thumb_index=True)
    m.locked = True
    thumb_sdk = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))["thumb_flex"]
    thumb_physical = (100.0 - thumb_sdk) if flip_thumb else thumb_sdk

    ref = CurlMapper(side=side, alpha=1.0, tuning=bounds, couple_thumb_index=True)
    ref.locked = True
    ref_physical = dict(zip(L6_SDK_ORDER, ref(_KP_INDEX_HALF)))["thumb_flex"]

    assert thumb_physical == pytest.approx(ref_physical, abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_shipped_config_coupling_is_flip_correct_on_both_hands(side):
    """End-to-end check of the REAL shipped config, for BOTH hands.

    Recomputes the expected coupled thumb independently -- undoing the index
    channel's flip to get physical openness, then re-applying thumb_flex's flip on
    the way out -- and derives the bounds from the resolved tuning rather than
    hardcoding them, so ongoing gain/pivot/couple_low tuning does not break it.

    What this actually guards is the SHIPPED CONFIG, per side: a `couple_low`
    accidentally moved into one `left:`/`right:` section, or a `flip` added on a
    coupling channel, changes the expectation here and shows up as a failure. It
    does NOT by itself prove the code honours flips -- with no flip configured on
    `thumb_flex` (today's config) a flip-dropping regression still satisfies it.
    ``test_couple_flip_matrix_is_physically_identical_on_both_hands`` is the test
    with teeth for that; verified by deliberately dropping the flip and watching
    that one fail while this one passed.
    """
    tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=side) or {}
    m = CurlMapper(side=side, alpha=1.0, tuning=tuning, couple_thumb_index=True)
    m.locked = True
    out = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))
    tracked = dict(zip(L6_SDK_ORDER, m.last_unparked))

    thumb_low = float(tuning.get("thumb_flex", {}).get("couple_low", 0.0))
    index_low = float(tuning.get("index", {}).get("couple_low", 0.0))
    index_flipped = bool(tuning.get("index", {}).get("flip"))
    thumb_flipped = bool(tuning.get("thumb_flex", {}).get("flip"))

    # last_unparked is post-flip; undo the index flip to recover physical openness.
    idx_phys = (100.0 - tracked["index"]) if index_flipped else tracked["index"]
    idx = max(index_low, idx_phys)
    span = 100.0 - index_low
    u = 1.0 if span <= 0.0 else min(max((idx - index_low) / span, 0.0), 1.0)
    v = min(max(thumb_low + u * (100.0 - thumb_low), 0.0), 100.0)
    expected_thumb = (100.0 - v) if thumb_flipped else v

    assert out["thumb_flex"] == pytest.approx(expected_thumb, abs=1e-6)
    if index_low > 0.0:
        expected_index = (100.0 - idx) if index_flipped else idx
        assert out["index"] == pytest.approx(expected_index, abs=1e-6)


def test_couple_degenerate_window_pins_the_thumb_open():
    """index_low == 100 leaves an empty working window (the index cannot move).
    That must not divide by zero -- the index is pinned fully open, so the thumb
    is too, which is the safe direction (no closure, no jam)."""
    m = CurlMapper(alpha=1.0,
                   tuning={"thumb_flex": {"couple_low": 30.0}, "index": {"couple_low": 100.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert by_slot["index"] == pytest.approx(100.0, abs=1e-6)
    assert by_slot["thumb_flex"] == pytest.approx(100.0, abs=1e-6)


def test_couple_index_floor_absent_leaves_index_untouched():
    """With no floor on index, the index channel is commanded exactly as it tracks."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 15.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert by_slot["index"] == pytest.approx(0.0, abs=1e-6)
    assert by_slot["thumb_flex"] == pytest.approx(15.0, abs=1e-6)


@pytest.mark.parametrize(("locked", "enabled"), [(False, False), (False, True), (True, False)])
def test_couple_index_floor_inert_unless_locked_and_enabled(locked, enabled):
    """The floor is part of the coupling, so it only applies while the coupling
    itself is engaged -- normal tracking is never clamped."""
    m = CurlMapper(alpha=1.0, tuning=_COUPLE_BOTH, couple_thumb_index=enabled)
    m.locked = locked
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert by_slot["index"] == pytest.approx(0.0, abs=1e-6)


def test_couple_index_floor_is_applied_in_physical_space_then_flipped():
    """The floor means 'this finger never closes past X' -- a statement about
    physical openness -- so it is clamped pre-flip and then converted into the
    hand's SDK space, exactly like the coupled thumb value."""
    plain = CurlMapper(alpha=1.0, tuning=_COUPLE_BOTH, couple_thumb_index=True)
    flipped = CurlMapper(
        alpha=1.0,
        tuning={**_COUPLE_BOTH, "index": {"couple_low": 20.0, "flip": True}},
        couple_thumb_index=True)
    plain.locked = flipped.locked = True
    v_plain = dict(zip(L6_SDK_ORDER, plain(_KP_INDEX_SHUT)))["index"]
    v_flipped = dict(zip(L6_SDK_ORDER, flipped(_KP_INDEX_SHUT)))["index"]
    assert v_plain == pytest.approx(20.0, abs=1e-6)
    assert v_flipped == pytest.approx(100.0 - 20.0, abs=1e-6)


def test_couple_index_floor_does_not_corrupt_the_index_ema():
    """Clamping happens post-EMA, so the index filter keeps tracking the real
    finger and unlocking resumes without a jump."""
    never = CurlMapper(alpha=0.4, tuning=_COUPLE_BOTH, couple_thumb_index=True)
    toggled = CurlMapper(alpha=0.4, tuning=_COUPLE_BOTH, couple_thumb_index=True)
    toggled.locked = True
    for _ in range(3):
        never(_KP_INDEX_SHUT)
        toggled(_KP_INDEX_SHUT)
    toggled.locked = False
    assert dict(zip(L6_SDK_ORDER, toggled(_KP_INDEX_SHUT)))["index"] == pytest.approx(
        dict(zip(L6_SDK_ORDER, never(_KP_INDEX_SHUT)))["index"], abs=1e-9)


def test_couple_index_floor_clipped_when_out_of_range():
    """An out-of-range floor must not escape to set_angles."""
    m = CurlMapper(alpha=1.0,
                   tuning={"thumb_flex": {"couple_low": 15.0}, "index": {"couple_low": 150.0}},
                   couple_thumb_index=True)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_INDEX_SHUT)))
    assert 0.0 <= by_slot["index"] <= 100.0
    assert 0.0 <= by_slot["thumb_flex"] <= 100.0


def test_couple_index_floor_exposed_for_the_readout():
    """The floor is surfaced so the console can mark the index as not-tracking."""
    assert CurlMapper(alpha=1.0, tuning=_COUPLE_BOTH).couple_index_floor == 20.0
    assert CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": 15.0}}).couple_index_floor is None


@pytest.mark.parametrize("low", [150.0, -20.0])
def test_couple_output_clipped_for_out_of_range_couple_low(low):
    """couple_low is not range-checked at load, so the coupled write must clip --
    every other write into `out` does, and this one feeds set_angles directly."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"couple_low": low}},
                   couple_thumb_index=True)
    m.locked = True
    assert 0.0 <= dict(zip(L6_SDK_ORDER, m(_KP_INDEX_HALF)))["thumb_flex"] <= 100.0


# -- MRP group (middle/ring/pinky commanded the median while locked) ------------ #

# Three genuinely distinct MRP postures, so the group tests exercise three
# distinct tracked values rather than a degenerate tie. Guarded by
# test_mrp_spread_fixture_has_three_distinct_readings below, since a
# collapsed fixture would make several of the tests below pass regardless of
# whether the implementation is actually correct.
_KP_MRP_SPREAD = _curl_kp_each({"middle": 20.0, "ring": 50.0, "pinky": 80.0})


def test_mrp_spread_fixture_has_three_distinct_readings():
    """Guards _KP_MRP_SPREAD itself: if it ever collapsed onto a tie, the
    tests below it (equalizes.../value_is_the_median...) would keep passing
    without actually exercising three distinct inputs. Pin the three tracked
    readings pairwise apart directly, so that kind of fixture rot fails
    loudly here instead of silently everywhere else."""
    tracked = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_MRP_SPREAD)))
    m, r, p = tracked["middle"], tracked["ring"], tracked["pinky"]
    assert m != pytest.approx(r, abs=1e-3)
    assert r != pytest.approx(p, abs=1e-3)
    assert m != pytest.approx(p, abs=1e-3)


def test_mrp_group_equalizes_the_three_channels_when_locked():
    """While locked, middle/ring/pinky are commanded ONE shared group value --
    all three outputs come out identical, even though the operator's actual
    fingers are at three different postures."""
    m = CurlMapper(alpha=1.0)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    assert by_slot["middle"] == pytest.approx(by_slot["ring"], abs=1e-9)
    assert by_slot["ring"] == pytest.approx(by_slot["pinky"], abs=1e-9)


def test_mrp_group_value_is_the_median_of_the_three_tracked_values():
    """Pins the exact formula: the group value is the median of the three
    fingers' own tracked (physical-openness) values, read off an identically
    constructed unlocked reference mapper."""
    m = CurlMapper(alpha=1.0)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))

    reference = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_MRP_SPREAD)))
    expected = float(np.median([reference["middle"], reference["ring"], reference["pinky"]]))
    for slot in ("middle", "ring", "pinky"):
        assert by_slot[slot] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("couple", [False, True])
def test_mrp_group_is_unconditional_and_ignores_couple_thumb_index(couple):
    """The group has NO opt-in of its own: `locked` alone drives it. Pinned
    explicitly because ``couple_thumb_index`` defaults to True in this package,
    so every other MRP test here runs with the thumb pinch enabled -- a
    regression that nested the group inside the pinch's ``if`` would satisfy
    all of them. With the pinch OFF the three channels must still collapse onto
    the same median.
    """
    m = CurlMapper(alpha=1.0, couple_thumb_index=couple)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))

    reference = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_MRP_SPREAD)))
    expected = float(np.median([reference["middle"], reference["ring"], reference["pinky"]]))
    for slot in ("middle", "ring", "pinky"):
        assert by_slot[slot] == pytest.approx(expected, abs=1e-6)


def test_mrp_group_inert_when_unlocked():
    """locked=False must leave the MRP channels byte-identical to a plain
    (never-locked) mapper -- the group is entirely a locked-mode behaviour."""
    m = CurlMapper(alpha=1.0)
    m.locked = False
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    plain = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_MRP_SPREAD)))
    assert by_slot == plain


# Pinky saturates at the 0 floor once its curl passes ~78 deg on this fixture
# (middle=20/ring=50 held fixed), so a perturbation that goes anywhere past
# that point would leave pinky's own reading UNCHANGED and prove nothing --
# the pair below (60 -> 70 deg) was picked to stay off that rail at both
# ends. Measured: pinky's own tracked reading moves ~32.8 -> ~13.4 while
# staying the extremum (below ring's ~58.3 and middle's ~88.2 throughout).
_KP_MRP_OUTLIER_BASE = _curl_kp_each({"middle": 20.0, "ring": 50.0, "pinky": 60.0})
_KP_MRP_OUTLIER_MOVED = _curl_kp_each({"middle": 20.0, "ring": 50.0, "pinky": 70.0})


def test_mrp_group_rejects_a_single_outlier_channel():
    """THE WHY-MEDIAN DOCUMENTATION TEST: two frames differing only in the
    most-extreme MRP channel (pinky, the most-closed of the three in both
    frames, moved further closed but kept off the 0 floor -- see
    _KP_MRP_OUTLIER_BASE/_MOVED above) must give the SAME group value, since
    the median reads off the untouched middle value (ring) and is blind to
    how far the extremum moves.

    Self-guarded against the vacuous version of this test (an unmoved input
    trivially gives an unmoved output, discriminating nothing): asserts the
    perturbed channel's own tracked reading (from last_unparked) really did
    move, AND that the mean of the three tracked values would therefore have
    moved too -- proving a mean-based group is not what's being measured --
    before checking that the actual (median-based) group value did not move.
    """
    m1 = CurlMapper(alpha=1.0)
    m1.locked = True
    out1 = dict(zip(L6_SDK_ORDER, m1(_KP_MRP_OUTLIER_BASE)))
    tracked1 = dict(zip(L6_SDK_ORDER, m1.last_unparked))

    m2 = CurlMapper(alpha=1.0)
    m2.locked = True
    out2 = dict(zip(L6_SDK_ORDER, m2(_KP_MRP_OUTLIER_MOVED)))
    tracked2 = dict(zip(L6_SDK_ORDER, m2.last_unparked))

    # The perturbation actually moved pinky's own reading...
    assert tracked1["pinky"] != pytest.approx(tracked2["pinky"], abs=1e-3)
    # ...and therefore would have moved a MEAN-based group too...
    mean1 = (tracked1["middle"] + tracked1["ring"] + tracked1["pinky"]) / 3.0
    mean2 = (tracked2["middle"] + tracked2["ring"] + tracked2["pinky"]) / 3.0
    assert mean1 != pytest.approx(mean2, abs=1e-3)
    # ...yet the actual (median-based) group value does not move at all.
    for slot in ("middle", "ring", "pinky"):
        assert out1[slot] == pytest.approx(out2[slot], abs=1e-9)


@pytest.mark.parametrize(
    "angles",
    [
        {"middle": 20.0, "ring": 50.0, "pinky": 80.0},
        {"middle": 0.0, "ring": 90.0, "pinky": 45.0},
        {"middle": 75.0, "ring": 10.0, "pinky": 60.0},
        {"middle": 30.0, "ring": 30.0, "pinky": 30.0},
    ],
)
def test_mrp_group_is_bounded_by_the_three_tracked_values(angles):
    """Safety invariant: median(a, b, c) is always within [min(a, b, c),
    max(a, b, c)] of the same three values, so the group can never command a
    hardware pose those channels could not already reach on their own.

    The three-are-equal assertion is what gives this teeth: the bound alone is
    satisfied trivially by three channels each emitting their OWN tracked value
    (i.e. by no group at all), so without it this test would still pass with
    the group removed. Together they say "one shared value, and that value is
    in range".
    """
    kp = _curl_kp_each(angles)
    m = CurlMapper(alpha=1.0)
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(kp)))

    reference = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(kp)))
    values = [reference["middle"], reference["ring"], reference["pinky"]]
    lo, hi = min(values), max(values)
    for slot in ("middle", "ring", "pinky"):
        assert lo - 1e-9 <= by_slot[slot] <= hi + 1e-9
    assert by_slot["middle"] == pytest.approx(by_slot["ring"], abs=1e-9)
    assert by_slot["ring"] == pytest.approx(by_slot["pinky"], abs=1e-9)


def test_mrp_group_applies_each_channels_flip():
    """A synthetic per-channel flip on ONE MRP channel: that channel emits the
    complement of the group value, while its unflipped groupmates emit the
    group value directly -- the flip is applied through EACH channel's own
    flip, not a single shared one."""
    m = CurlMapper(alpha=1.0, tuning={"middle": {"flip": True}})
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    g = by_slot["ring"]
    assert by_slot["pinky"] == pytest.approx(g, abs=1e-9)
    assert by_slot["middle"] == pytest.approx(100.0 - g, abs=1e-6)


def test_mrp_group_reads_before_the_channels_own_flips():
    """The group is computed from `pre` (physical openness, pre-flip), so
    flipping one channel's OWN output must not move the group value itself --
    only that channel's write of it."""
    plain = CurlMapper(alpha=1.0)
    ring_flipped = CurlMapper(alpha=1.0, tuning={"ring": {"flip": True}})
    plain.locked = ring_flipped.locked = True
    g_plain = dict(zip(L6_SDK_ORDER, plain(_KP_MRP_SPREAD)))["middle"]
    g_via_flipped_mapper = dict(zip(L6_SDK_ORDER, ring_flipped(_KP_MRP_SPREAD)))["pinky"]
    assert g_via_flipped_mapper == pytest.approx(g_plain, abs=1e-6)


def test_mrp_group_wins_over_a_park_on_an_mrp_channel():
    """The group is applied AFTER the park loop, so the narrower opt-in wins
    when an MRP channel somehow carries both -- exactly like the thumb
    coupling's precedence over a park on thumb_flex."""
    m = CurlMapper(alpha=1.0, tuning={"ring": {"park": 15.0}})
    m.locked = True
    by_slot = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    assert by_slot["ring"] == pytest.approx(by_slot["middle"], abs=1e-9)
    assert by_slot["ring"] != pytest.approx(15.0, abs=1.0)


def test_mrp_group_leaves_last_unparked_tracking_the_real_fingers():
    """last_unparked snapshots BEFORE the group is applied, so a UI can still
    show the operator's three actual fingers underneath the shared group
    command."""
    m = CurlMapper(alpha=1.0)
    m.locked = True
    out = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    tracked = dict(zip(L6_SDK_ORDER, m.last_unparked))

    reference = dict(zip(L6_SDK_ORDER, CurlMapper(alpha=1.0)(_KP_MRP_SPREAD)))
    for slot in ("middle", "ring", "pinky"):
        assert tracked[slot] == pytest.approx(reference[slot], abs=1e-6)
    # the group actually changed at least one of the three in the output.
    assert any(
        out[slot] != pytest.approx(tracked[slot], abs=1e-6) for slot in ("middle", "ring", "pinky")
    )


def test_mrp_group_does_not_corrupt_ema_so_unlock_has_no_jump():
    """The group write lands post-EMA, so each channel's filter keeps tracking
    the real finger underneath: unlocking resumes exactly where a
    never-locked mapper would be, with no jump."""
    never = CurlMapper(alpha=0.4)
    toggled = CurlMapper(alpha=0.4)
    toggled.locked = True
    for _ in range(3):  # same frames through both; only `toggled` is locked
        never(_KP_MRP_SPREAD)
        toggled(_KP_MRP_SPREAD)
    toggled.locked = False
    never_out = dict(zip(L6_SDK_ORDER, never(_KP_MRP_SPREAD)))
    toggled_out = dict(zip(L6_SDK_ORDER, toggled(_KP_MRP_SPREAD)))
    for slot in ("middle", "ring", "pinky"):
        assert toggled_out[slot] == pytest.approx(never_out[slot], abs=1e-9)


def test_mrp_group_does_not_touch_thumb_or_index():
    """The group's channel set (middle/ring/pinky) is disjoint from the thumb
    coupling's (thumb_flex/index): with couple_thumb_index also enabled,
    varying only the MRP fingers' postures (holding index/thumb fixed) must
    not move thumb_flex, thumb_abd or index at all."""
    kp_a = _curl_kp_each({"index": 45.0, "middle": 20.0, "ring": 50.0, "pinky": 80.0})
    kp_b = _curl_kp_each({"index": 45.0, "middle": 80.0, "ring": 20.0, "pinky": 50.0})

    m_a = CurlMapper(alpha=1.0, couple_thumb_index=True)
    m_a.locked = True
    out_a = dict(zip(L6_SDK_ORDER, m_a(kp_a)))

    m_b = CurlMapper(alpha=1.0, couple_thumb_index=True)
    m_b.locked = True
    out_b = dict(zip(L6_SDK_ORDER, m_b(kp_b)))

    for slot in ("thumb_flex", "thumb_abd", "index"):
        assert out_a[slot] == pytest.approx(out_b[slot], abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_shipped_config_mrp_group_is_flip_correct_on_both_hands(side):
    """End-to-end check of the REAL shipped config's MRP group, for BOTH
    hands: recomputes the expected group value independently -- undoing each
    channel's own flip to get physical openness, taking the median, then
    re-applying each channel's own flip on the way out -- and derives the
    flip flags from the resolved tuning rather than hardcoding them, so
    ongoing tuning changes do not silently break it, mirroring
    ``test_shipped_config_coupling_is_flip_correct_on_both_hands``.
    """
    tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=side) or {}
    m = CurlMapper(side=side, alpha=1.0, tuning=tuning)
    m.locked = True
    out = dict(zip(L6_SDK_ORDER, m(_KP_MRP_SPREAD)))
    tracked = dict(zip(L6_SDK_ORDER, m.last_unparked))

    mrp_slots = ("middle", "ring", "pinky")
    flipped = {slot: bool(tuning.get(slot, {}).get("flip")) for slot in mrp_slots}
    physical = {
        slot: (100.0 - tracked[slot]) if flipped[slot] else tracked[slot] for slot in mrp_slots
    }
    expected_group = float(np.median([physical[slot] for slot in mrp_slots]))
    for slot in mrp_slots:
        expected = (100.0 - expected_group) if flipped[slot] else expected_group
        assert out[slot] == pytest.approx(expected, abs=1e-6)


# -- static home gesture (home_gesture() + tuning's home_gesture key) ------------ #

# The default thumbs-up in PHYSICAL-openness terms: thumb extended + spread,
# the other four fingers curled into a fist. Every case below must recover
# exactly this, whatever hand/flip combination produced the raw SDK numbers.
_HOME_GESTURE_DEFAULT_OPENNESS = {
    "thumb_flex": 100.0, "thumb_abd": 100.0,
    "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0,
}


def test_home_gesture_default_left_hand_no_tuning():
    """No tuning, left hand: the default thumbs-up, in L6_SDK_ORDER order
    (thumb_flex, thumb_abd, index, middle, ring, pinky)."""
    mapper = CurlMapper(side="left", alpha=1.0)
    assert mapper.home_gesture() == pytest.approx(
        [100.0, 100.0, 0.0, 0.0, 0.0, 0.0], abs=1e-9
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    ("flip_thumb_flex", "flip_thumb_abd"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_home_gesture_flip_matrix_is_physically_identical_on_both_hands(
    side, flip_thumb_flex, flip_thumb_abd
):
    """THE CRITICAL GUARD: whichever hand and whatever the flip combination on
    thumb_flex/thumb_abd, home_gesture() must encode the SAME PHYSICAL pose --
    a flip changes only the SDK encoding of that pose, never the pose itself.

    Un-flip the returned SDK values back to openness using THIS mapper's own
    flip set and compare against the fixed openness every one of the 8 cases
    (2 sides x 4 flip combos) must recover -- so they are all identical to
    each other by transitivity too. This is the test that catches a later
    change that makes ``home_gesture()`` bypass the flip the way ``park``
    deliberately does.
    """
    tuning = {
        "thumb_flex": {"flip": flip_thumb_flex},
        "thumb_abd": {"flip": flip_thumb_abd},
    }
    mapper = CurlMapper(side=side, alpha=1.0, tuning=tuning)
    sdk = dict(zip(L6_SDK_ORDER, mapper.home_gesture()))

    flipped_slots = {
        slot for slot, flipped in
        (("thumb_flex", flip_thumb_flex), ("thumb_abd", flip_thumb_abd)) if flipped
    }
    openness = {
        slot: (100.0 - v if slot in flipped_slots else v) for slot, v in sdk.items()
    }
    assert openness == _HOME_GESTURE_DEFAULT_OPENNESS


def test_shipped_config_home_gesture_matches_hardware_on_both_hands():
    """End-to-end check of the REAL shipped config's home_gesture, per hand.

    thumb_abd is the one channel where the shipped config deliberately does NOT
    mirror: `right: thumb_abd: {flip: true, home_gesture: 0}` pre-flips the
    right hand's value so that, after home_gesture() puts it through the flip,
    BOTH hands command SDK 100 -- the direction confirmed to point the thumb up
    on hardware (2026-07-29). The `flip` still applies to the tracked metric
    during teleop; only this static pose opts out of the mirror. The left hand
    is untouched by that override.

    Pins the actual expected numbers so an edit to curl_tuning.yml that changes
    either hand's homing thumbs-up fails loudly.
    """
    left = CurlMapper(side="left", alpha=1.0,
                      tuning=resolve_tuning(DEFAULT_TUNING_PATH, side="left"))
    right = CurlMapper(side="right", alpha=1.0,
                       tuning=resolve_tuning(DEFAULT_TUNING_PATH, side="right"))
    left_sdk = dict(zip(L6_SDK_ORDER, left.home_gesture()))
    right_sdk = dict(zip(L6_SDK_ORDER, right.home_gesture()))

    assert left_sdk == _HOME_GESTURE_DEFAULT_OPENNESS  # left has no flips configured
    assert right_sdk == {
        "thumb_flex": 100.0, "thumb_abd": 100.0,  # pre-flipped 0, flipped back out
        "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0,
    }
    # The right hand's abd is NOT the flip-mirror of the left's -- that is the
    # point of the override. Both hands land on the same raw SDK value instead.
    assert right_sdk["thumb_abd"] == pytest.approx(left_sdk["thumb_abd"], abs=1e-9)


def test_home_gesture_unaffected_by_locked():
    """home_gesture() must not depend on self.locked."""
    kp = _curl_kp(("thumb",), angle_deg=45)
    m = CurlMapper(alpha=1.0)
    before = m.home_gesture()
    m.locked = True
    m(kp)
    assert m.home_gesture() == pytest.approx(before, abs=1e-9)


def test_home_gesture_unaffected_by_configured_park():
    """home_gesture() must not depend on self._parks, even while locked."""
    m = CurlMapper(alpha=1.0, tuning={"thumb_flex": {"park": 15.0}})
    m.locked = True
    reference = CurlMapper(alpha=1.0).home_gesture()
    assert m.home_gesture() == pytest.approx(reference, abs=1e-9)


def test_home_gesture_unaffected_by_couple_thumb_index():
    """home_gesture() must not depend on the thumb<-index coupling."""
    kp = _curl_kp(("index",), angle_deg=45)
    m = CurlMapper(alpha=1.0, couple_thumb_index=True)
    m.locked = True
    m(kp)
    reference = CurlMapper(alpha=1.0).home_gesture()
    assert m.home_gesture() == pytest.approx(reference, abs=1e-9)


def test_home_gesture_does_not_disturb_a_following_call():
    """Calling home_gesture() must be pure: it must not perturb the EMA state
    (self._last) that a following __call__(kp) resumes from."""
    kp = _curl_kp(("index",), angle_deg=45)
    reference = CurlMapper(alpha=0.4)
    reference(kp)
    expected = reference(kp)

    probed = CurlMapper(alpha=0.4)
    probed(kp)
    probed.home_gesture()
    probed.home_gesture()
    assert probed(kp) == pytest.approx(expected, abs=1e-9)


def test_home_gesture_tuning_override_is_honoured():
    """An optional per-channel `home_gesture` tuning key overrides just that
    channel's default value."""
    m = CurlMapper(alpha=1.0, tuning={"index": {"home_gesture": 42.0}})
    by_slot = dict(zip(L6_SDK_ORDER, m.home_gesture()))
    expected = dict(_HOME_GESTURE_DEFAULT_OPENNESS, index=42.0)
    assert by_slot == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("raw", "clipped"), [(150.0, 100.0), (-20.0, 0.0)])
def test_home_gesture_tuning_override_is_clipped_to_range(raw, clipped):
    """An out-of-range home_gesture override is clipped to [0, L6_OPEN] at
    ingest, mirroring couple_low's clamp."""
    m = CurlMapper(alpha=1.0, tuning={"pinky": {"home_gesture": raw}})
    by_slot = dict(zip(L6_SDK_ORDER, m.home_gesture()))
    assert by_slot["pinky"] == pytest.approx(clipped, abs=1e-9)


def test_home_gesture_override_still_passes_through_flip():
    """A tuning-overridden home_gesture value is still a physical/tracked
    quantity like the default, so a flipped channel emits its complement, not
    the raw override value."""
    m = CurlMapper(
        alpha=1.0, tuning={"thumb_abd": {"home_gesture": 30.0, "flip": True}}
    )
    by_slot = dict(zip(L6_SDK_ORDER, m.home_gesture()))
    assert by_slot["thumb_abd"] == pytest.approx(70.0, abs=1e-9)
