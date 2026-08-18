"""TDD validators for prehensile/curlmap.py's tuning-driven per-channel knobs.

Pure-synthetic, hardware-free: CurlMapper's constructor consumes a plain
``tuning`` dict (the shape ``prehensile.tuning.resolve_tuning`` produces)
without needing a glove, retargeter, or L6.
"""

import math

import numpy as np
import pytest

from prehensile.command import L6_SDK_ORDER
from prehensile.curlmap import CurlMapper
from prehensile.profiles import GLOVES


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
