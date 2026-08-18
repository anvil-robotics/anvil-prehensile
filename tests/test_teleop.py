"""TDD validators for prehensile/teleop.py (UDCap -> L6 teleop port).

Headless + hardware-free, and deliberately also DEPENDENCY-free: nothing here
imports prehensile.retarget, so the whole module collects and runs on a bare
`pip install prehensile` (no `research` extra) -- which is what lets CI's lean
job, the pull-request gate, actually run these tests. The three teleop tests
that do need a real L6Retargeter live in tests/test_teleop_retarget.py; keep
this file's import list clean of the heavy stack. The --live L6 path, its 3-2-1
countdown, and the CAN error handling are deferred to hardware acceptance with
the user (`uv run python -m prehensile.teleop --live`).
"""

import itertools

import pytest
from prehensile import fk
from prehensile.curlmap import CurlMapper
from prehensile.profiles import GLOVES, HANDS
from prehensile import teleop


def test_glove_and_hand_profiles():
    """GLOVES/HANDS registries wire up both gloves and both hand sides."""
    assert set(GLOVES) >= {"wuji", "udcap"}
    assert GLOVES["wuji"].invert_flex is False
    assert GLOVES["udcap"].invert_flex is True
    # curl-map thumb_abd sense (per-glove, hardware-observed): under the
    # palm-plane abduction metric UDCap inverts and Wuji does not.
    assert GLOVES["wuji"].abd_invert is False
    assert GLOVES["udcap"].abd_invert is True
    assert HANDS["left"].config == "real_hand_left.yml"
    assert HANDS["right"].side == "right"

    # Constructing (not entering) a udcap source is hardware-free: it just
    # binds a UDP socket, matching UDCapSource's own shape.
    src = GLOVES["udcap"].build_source("left", 5599)
    try:
        assert callable(src.poll)
        assert hasattr(src, "__enter__") and hasattr(src, "__exit__")
    finally:
        src.close()


# -- Phase 2: bButton park-lock toggle + console display (_fmt / loop()) --------- #


def test_fmt_with_no_parked_channels_is_unchanged():
    """_fmt(angles) with an empty (or omitted) marker arg is the bare per-channel readout."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    bare = ("thumb_flex= 82.3  thumb_abd= 47.1  index= 91.0  "
            "middle= 88.4  ring= 90.2  pinky= 89.7")
    assert teleop._fmt(angles) == bare
    assert teleop._fmt(angles, ()) == bare


def test_fmt_with_parked_channels_appends_marker():
    """A non-empty parked-channel tuple appends '  [PARKED <names>]' verbatim."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, ("thumb_abd",))
    assert line == ("thumb_flex= 82.3  thumb_abd= 47.1  index= 91.0  middle= 88.4  "
                     "ring= 90.2  pinky= 89.7  [PARKED thumb_abd]")


def test_fmt_joins_multiple_parked_channels_with_commas():
    """Multiple parked names are comma-joined inside the single marker."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, ("thumb_abd", "index"))
    assert line.endswith("  [PARKED thumb_abd, index]")


# loop() is an infinite `while True`; these fakes let it run for an exact,
# scripted number of frames and then stop it via a sentinel exception instead
# of mocking loop() itself, so the button/lock/display logic under test is
# genuinely exercised end-to-end (real CurlMapper, real loop() body).


class _StopScript(Exception):
    """Raised by a fake source once its scripted frames are exhausted, to
    unwind out of loop()'s ``while True`` from inside a test."""


class _ScriptedButtonSource:
    """Fake glove source: replays a scripted ``(kp, bButton)`` sequence, one
    pair per ``poll()`` call, then raises ``_StopScript``. Mirrors the real
    ``UDCapSource`` contract that loop() relies on: ``self.bButton`` reflects
    the newest polled frame."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.bButton = False

    def poll(self):
        if self._i >= len(self._script):
            raise _StopScript
        kp, pressed = self._script[self._i]
        self._i += 1
        self.bButton = pressed
        return kp


class _NoButtonSource:
    """Fake glove source with NO ``bButton`` attribute at all, like the real
    ``WujiSource`` -- loop() must use ``getattr(..., False)`` rather than
    assume the attribute exists."""

    def __init__(self, kps):
        self._kps = list(kps)
        self._i = 0

    def poll(self):
        if self._i >= len(self._kps):
            raise _StopScript
        kp = self._kps[self._i]
        self._i += 1
        return kp


class _RecordingSink:
    """``sink`` fake: records every list of angles it is called with."""

    def __init__(self):
        self.calls: list[list[float]] = []

    def __call__(self, angles) -> None:
        self.calls.append(list(angles))


def _identity_kp():
    """The same neutral-pose (21,3) keypoint frame the retarget tests above
    use. Under a plain ``CurlMapper(side="left")`` it settles to a fully-open
    reading with thumb_abd pinned at its 0.0 (tucked) bound -- a large,
    unambiguous gap from any park value used below, so comparisons against a
    parked value are never a rounding coincidence (verified: angles ==
    [100.0, 0.0, 100.0, 100.0, 100.0, 100.0])."""
    return fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)


def test_loop_bbutton_rising_edge_toggles_lock_on():
    """A press (False -> True) flips mapper.locked off -> on."""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}})
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    assert mapper.locked is False
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is True


def test_loop_bbutton_held_toggles_only_once():
    """Holding the button across several frames must NOT re-toggle each frame."""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}})
    kp = _identity_kp()
    source = _ScriptedButtonSource([(kp, True), (kp, True), (kp, True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is True  # one rising edge only, not 3


def test_loop_bbutton_second_press_toggles_lock_off():
    """press -> release -> press flips on, then back off.

    Also checks the MIDDLE frame's sink call (thumb_abd forced to its 5.0
    park value) to prove the mapper really was locked in between the two
    presses -- otherwise a do-nothing implementation that never locks at all
    would also (trivially, wrongly) satisfy the final `locked is False`.
    """
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}})
    kp = _identity_kp()
    source = _ScriptedButtonSource([(kp, True), (kp, False), (kp, True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is False
    assert len(sink.calls) == 3
    assert sink.calls[1][1] == 5.0  # thumb_abd; locked during the middle (released-but-held) frame


def test_loop_source_without_bbutton_never_locks_and_does_not_raise():
    """A Wuji-like source with no bButton attribute is a silent no-op (getattr default)."""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}})
    kp = _identity_kp()
    source = _NoButtonSource([kp, kp, kp])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is False
    assert len(sink.calls) == 3


def test_loop_locked_display_shows_tracked_value_sink_gets_parked(capsys):
    """While locked: the sink gets the real (parked) command, but the console
    readout shows the live tracked value underneath plus the marker."""
    # couple_thumb_index=False: this test is isolated to park behaviour (now
    # that coupling defaults on, leaving it enabled would also print a COUPLED
    # segment and break the exact bracket match below). The GROUP segment is
    # there regardless -- the MRP grasp group is unconditional under the lock.
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}},
                        couple_thumb_index=False)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)

    assert mapper.locked is True
    thumb_abd_idx = 1  # L6_SDK_ORDER / JOINTS: [thumb_flex, thumb_abd, index, middle, ring, pinky]
    assert sink.calls[-1][thumb_abd_idx] == 5.0  # hardware got the literal park value
    tracked = mapper.last_unparked[thumb_abd_idx]
    assert tracked == 0.0  # the identity pose's genuine tracked reading (see _identity_kp)

    out = capsys.readouterr().out
    assert "[PARKED thumb_abd  GROUP middle=ring=pinky]" in out
    assert f"thumb_abd={tracked:5.1f}" in out       # displayed: the tracked value...
    assert f"thumb_abd={5.0:5.1f}" not in out        # ...never the parked one


def test_loop_unlocked_display_has_no_marker_even_with_a_park_configured(capsys):
    """A configured park value alone must not show the marker -- only actually
    being locked does. (Distinct from the mapper=None case in
    tests/test_teleop_retarget.py: here the mapper exists and DOES have a parked
    channel, it's just never engaged.)"""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}})
    assert mapper.parked_channels == ("thumb_abd",)
    source = _ScriptedButtonSource([(_identity_kp(), False)])  # bButton never pressed
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is False
    assert "[PARKED" not in capsys.readouterr().out


def test_loop_redraw_padding_erases_locked_marker_residue(monkeypatch, capsys):
    """Unlocking must not leave '[PARKED thumb_abd]' stranded on screen: each
    redraw has to be padded to at least the previous redraw's width so a
    shrinking line fully overwrites the longer one.

    Drives loop() with a fake, deterministically-advancing clock (instead of
    real sleeping) so all three scripted frames land on the print side of the
    80ms console throttle every time -- no real time.sleep, no flakiness.
    loop() reads time.monotonic() twice per iteration (t0, then the
    end-of-loop dt check); a plain 0.1s-stepping counter feeds both calls, so
    consecutive frames are always >0.08s apart from the caller's perspective.
    """
    # couple_thumb_index=False: isolate this test to the park marker (coupling
    # defaults on and would add its own "COUPLED ..." segment to every line).
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}},
                        couple_thumb_index=False)
    kp = _identity_kp()
    fake_clock = itertools.count(1.0, 0.1)
    monkeypatch.setattr(teleop.time, "monotonic", lambda: next(fake_clock))
    monkeypatch.setattr(teleop.time, "sleep", lambda _dt: None)

    # press (lock on) -> release (still locked) -> press (lock off).
    source = _ScriptedButtonSource([(kp, True), (kp, False), (kp, True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is False

    segments = capsys.readouterr().out.split("\r")[1:]  # drop the empty pre-"\r" piece
    assert len(segments) == 3  # one redraw per frame -- confirms the fake clock design
    locked_line, still_locked_line, unlocked_line = segments
    assert "[PARKED thumb_abd  GROUP middle=ring=pinky]" in locked_line
    assert "[PARKED thumb_abd  GROUP middle=ring=pinky]" in still_locked_line
    assert "[PARKED" not in unlocked_line
    # The old fixed 4-space suffix could strand the marker on screen; the fix
    # must pad the shorter unlocked redraw out to fully cover the prior one.
    assert len(unlocked_line) >= len(still_locked_line.rstrip())


# -- thumb<-index coupling (--thumb-index-couple) -------------------------------- #

# A 0.5 gain on `index` pulls the identity pose's index off its 100.0 ceiling to
# 75.0 (out = pivot + (raw - pivot) * gain = 50 + 50*0.5), while leaving
# thumb_flex's own tracked reading at 100.0. That gap is what makes "the sink got
# the coupled value, the screen showed the tracked one" a real assertion rather
# than a coincidence of two channels both sitting at 100.
_COUPLE_TUNING = {"thumb_flex": {"couple_low": 30.0}, "index": {"gain": 0.5}}
_COUPLED_THUMB = 30.0 + (75.0 / 100.0) * 70.0  # == 82.5


def test_fmt_with_coupled_only_appends_coupled_marker():
    """coupled alone renders its own segment, with no PARKED text."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, (), coupled=("thumb_flex", "index"))
    assert line.endswith("  [COUPLED thumb_flex<-index]")
    assert "PARKED" not in line


def test_fmt_composes_parked_and_coupled_in_one_bracket():
    """Both markers share a single bracket, space-separated."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, ("thumb_abd",), coupled=("thumb_flex", "index"))
    assert line.endswith("  [PARKED thumb_abd  COUPLED thumb_flex<-index]")


def test_loop_coupled_sink_gets_coupled_value_display_shows_tracked(capsys):
    """While coupled, the hand is commanded the index-derived thumb value but the
    readout keeps showing the operator's real tracked thumb."""
    mapper = CurlMapper(side="left",
                        tuning={"thumb_abd": {"park": 5.0}, **_COUPLE_TUNING},
                        couple_thumb_index=True)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert mapper.locked is True
    assert sink.calls[0][2] == pytest.approx(75.0, abs=1e-6)             # index
    assert sink.calls[0][0] == pytest.approx(_COUPLED_THUMB, abs=1e-6)   # thumb_flex
    out = capsys.readouterr().out
    assert "COUPLED thumb_flex<-index" in out
    assert f"thumb_flex={100.0:5.1f}" in out                    # tracked...
    assert f"thumb_flex={_COUPLED_THUMB:5.1f}" not in out       # ...not the command


def test_loop_no_coupled_marker_when_coupling_disabled_at_construction(capsys):
    """Locking with couple_thumb_index=False keeps the existing park-only
    behaviour -- still a valid, directly-constructible configuration even
    though the CLI no longer has a flag to reach it."""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}},
                        couple_thumb_index=False)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    out = capsys.readouterr().out
    assert "[PARKED thumb_abd  GROUP middle=ring=pinky]" in out
    assert "COUPLED" not in out


def test_fmt_with_index_floor_appends_floor_segment():
    """A configured index floor gets its own marker segment, since the index's
    displayed number is then also not what is being commanded."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, (), coupled=("thumb_flex", "index"), floored=("index", 20.0))
    assert line.endswith("  [COUPLED thumb_flex<-index  FLOOR index=20]")


def test_fmt_with_grouped_only_appends_group_marker():
    """grouped alone renders its own segment, with no PARKED/COUPLED/FLOOR text."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, (), grouped=("middle", "ring", "pinky"))
    assert line.endswith("  [GROUP middle=ring=pinky]")
    assert "PARKED" not in line
    assert "COUPLED" not in line


def test_fmt_composes_all_markers_in_one_bracket():
    """All four markers share a single bracket, in PARKED -> COUPLED -> FLOOR ->
    GROUP order (GROUP last, since it's CurlMapper's last-applied step)."""
    angles = [82.3, 47.1, 91.0, 88.4, 90.2, 89.7]
    line = teleop._fmt(angles, ("thumb_abd",), coupled=("thumb_flex", "index"),
                       floored=("index", 20.0), grouped=("middle", "ring", "pinky"))
    assert line.endswith(
        "  [PARKED thumb_abd  COUPLED thumb_flex<-index  FLOOR index=20  GROUP middle=ring=pinky]"
    )


def test_loop_index_floor_clamps_sink_and_marks_the_readout(capsys):
    """With a floor on index, the sink gets the clamped index while the readout
    still shows the tracked one, flagged by a FLOOR segment."""
    mapper = CurlMapper(side="left",
                        tuning={"thumb_flex": {"couple_low": 15.0},
                                "index": {"couple_low": 20.0, "gain": 0.5}},
                        couple_thumb_index=True)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    # identity pose + 0.5 index gain -> tracked index 75.0, above the 20 floor,
    # so the floor is configured-but-not-biting; the marker must still appear.
    assert sink.calls[0][2] == pytest.approx(75.0, abs=1e-6)
    out = capsys.readouterr().out
    assert "FLOOR index=20" in out
    assert f"index={75.0:5.1f}" in out


def test_loop_no_floor_segment_when_index_floor_unset(capsys):
    """Coupling without an index floor renders only the COUPLED segment."""
    mapper = CurlMapper(side="left", tuning={"thumb_abd": {"park": 5.0}, **_COUPLE_TUNING},
                        couple_thumb_index=True)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    out = capsys.readouterr().out
    assert "COUPLED thumb_flex<-index" in out
    assert "FLOOR" not in out


def test_loop_coupled_channel_is_not_reported_as_parked(capsys):
    """A `park` can legitimately be set on thumb_flex (configs/curl_tuning.yml).
    When coupling also owns that channel the coupled value wins, so the readout
    must not claim the channel is PARKED."""
    mapper = CurlMapper(side="left",
                        tuning={"thumb_flex": {"park": 15.0, "couple_low": 30.0},
                                "index": {"gain": 0.5}},
                        couple_thumb_index=True)
    assert "thumb_flex" in mapper.parked_channels  # configuration still says so
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, None, None, sink, fps=1000.0, mapper=mapper)
    assert sink.calls[0][0] == pytest.approx(_COUPLED_THUMB, abs=1e-6)  # not the 15.0 park
    out = capsys.readouterr().out
    assert "COUPLED thumb_flex<-index" in out
    assert "PARKED thumb_flex" not in out


