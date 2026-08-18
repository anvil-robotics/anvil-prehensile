"""TDD validators for prehensile/wuji.py (Wuji SDK glove source).

SYNTHETIC-ONLY: no real wuji_sdk, no hardware. Every test builds fakes in code
that mimic just enough of the wuji_sdk shape (``sub.recv()`` -> frame|None,
``frame.joints`` -> 21 objects each with ``.pose.position`` -> length-3
sequence) to exercise ``read_latest_keypoints`` and ``WujiSource.poll()``
without ever importing or requiring ``wuji_sdk`` itself.

Covers:
  1. ``read_latest_keypoints`` drains a fake sub and returns the NEWEST frame.
  2. ``WujiSource.poll()`` contract with an injected fake sub (bypassing
     ``__enter__`` entirely, so no SdkManager/glove is ever touched).
  3. ``prehensile.wuji`` imports fine, and its numpy-only helpers still work,
     even when ``wuji_sdk`` is unimportable.
  4. ``wrist_center`` subtracts landmark 0.
"""

import importlib
import sys

import numpy as np

from prehensile.wuji import WujiSource, read_latest_keypoints, wrist_center


# --------------------------------------------------------------------------- #
# Fakes: mimic the tiny slice of the wuji_sdk shape we depend on, nothing more.
# --------------------------------------------------------------------------- #
class _FakePose:
    def __init__(self, position):
        self.position = position


class _FakeJoint:
    def __init__(self, position):
        self.pose = _FakePose(position)


class _FakeFrame:
    """21 joints, each at a constant offset ``value`` in all 3 axes."""

    def __init__(self, value):
        self.joints = [_FakeJoint([value, value, value]) for _ in range(21)]


class _FakeSub:
    """A subscription whose ``.recv()`` yields queued frames then None."""

    def __init__(self, frames):
        self._frames = list(frames)

    def recv(self):
        if not self._frames:
            return None
        return self._frames.pop(0)


# --------------------------------------------------------------------------- #
# 1. read_latest_keypoints: drain-latest semantics.
# --------------------------------------------------------------------------- #
def test_read_latest_keypoints_returns_newest_frame():
    sub = _FakeSub([_FakeFrame(1.0), _FakeFrame(2.0), _FakeFrame(3.0)])
    kp = read_latest_keypoints(sub)
    assert kp is not None
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_array_equal(kp, np.full((21, 3), 3.0, dtype=np.float32))


def test_read_latest_keypoints_none_when_no_frames_queued():
    sub = _FakeSub([])
    assert read_latest_keypoints(sub) is None


# --------------------------------------------------------------------------- #
# 2. WujiSource.poll() contract, with an injected fake sub (no __enter__).
# --------------------------------------------------------------------------- #
def test_wuji_source_poll_returns_frame_from_injected_sub():
    src = WujiSource()
    src._sub = _FakeSub([_FakeFrame(0.5), _FakeFrame(1.5)])
    kp = src.poll()
    assert kp is not None
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_array_equal(kp, np.full((21, 3), 1.5, dtype=np.float32))


def test_wuji_source_poll_none_for_empty_sub():
    src = WujiSource()
    src._sub = _FakeSub([])
    assert src.poll() is None


def test_wuji_source_close_is_safe_when_never_entered():
    src = WujiSource()
    src.close()  # must not raise even though __enter__ never ran


# --------------------------------------------------------------------------- #
# 3. Lazy-import proof: module (and its numpy-only helpers) survive wuji_sdk
#    being unimportable.
# --------------------------------------------------------------------------- #
def test_imports_without_wuji_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "wuji_sdk", None)  # makes `import wuji_sdk` raise ImportError
    sys.modules.pop("prehensile.wuji", None)
    mod = importlib.import_module("prehensile.wuji")
    assert hasattr(mod, "WujiSource") and hasattr(mod, "read_latest_keypoints")

    # The numpy-only helpers must still work with no wuji_sdk present.
    sub = _FakeSub([_FakeFrame(7.0)])
    kp = mod.read_latest_keypoints(sub)
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32

    # Re-import for real so later tests in this process see the normal module.
    sys.modules.pop("prehensile.wuji", None)
    importlib.import_module("prehensile.wuji")


# --------------------------------------------------------------------------- #
# 4. wrist_center subtracts landmark 0.
# --------------------------------------------------------------------------- #
def test_wrist_center_subtracts_landmark_zero():
    kp = np.array(
        [[1.0, 1.0, 1.0]] + [[2.0, 3.0, 4.0]] * 20, dtype=np.float32
    )
    centered = wrist_center(kp)
    np.testing.assert_array_equal(centered[0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(centered[1], [1.0, 2.0, 3.0])
