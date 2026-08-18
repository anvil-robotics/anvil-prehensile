"""Direct per-finger curl map: MediaPipe (21,3) keypoints -> per-channel
openness angles (0-100), one per channel of ``hand`` (``L6_HAND`` -- the L6's
6 channels -- by default; see ``prehensile.hand.HandDescriptor``).

Bypasses the dex_retargeting vector optimizer entirely for teleop. The L6 has
only one flex DOF per finger + a thumb abduction DOF.

Four-finger curl metric (scale-invariant chord ratio), per flex channel, chain
(0=wrist -> mcp -> pip -> dip -> tip):
    L_f    = sum of the chain's 4 segment lengths (constant; FK preserves bone length)
    r_f    = |kp[tip] - kp[wrist]| / L_f          # ~1.0 straight/open, lower when curled
    flex_f = clip((r_f - R_CLOSED[f]) / (R_OPEN[f] - R_CLOSED[f]), 0, 1) * 100   # 100=open

The THUMB does NOT use the chord ratio: it is short and mostly opposes/rotates
rather than bringing its tip toward the wrist, so that chord barely moves (weak,
mushy drive) and even folds back at high flexion. Instead the two thumb DOFs are
decoupled onto their own metrics:

  * thumb FLEXION = the inter-bone joint-bend angle (thumb MCP bend + IP bend of
    kp[1..4]): ~0 deg straight, growing monotonically as the thumb curls -- a
    large, well-conditioned signal that is invariant to where the thumb points.
    Mapped open(small bend)->100, closed(large bend)->0 via
    THUMB_FLEX_OPEN_DEG / THUMB_FLEX_CLOSED_DEG.

  * thumb ABDUCTION = the elevation of the thumb metacarpal (kp1->kp2) OUT of
    the palm plane (wrist, index-mcp, pinky-mcp): arcsin(unit(kp2-kp1) .
    palm_normal), signed. Palmar abduction lifts the thumb perpendicular to the
    palm, while flexion curls it ACROSS the palm (in-plane), so the out-of-plane
    component is far less flexion-confounded than an in-plane angle -- verified
    on real UDCap-left data (corr with flexion ~0.88 in-plane -> ~0.69
    out-of-plane, ~2.5x more flexion-independent signal). Mapped via ABD_TUCK /
    ABD_SPREAD, then optionally complemented by ``abd_invert``.

Both thumb metrics are best set by the dedicated live calibration gestures
(THUMB EXTENDED/CURL for flexion, THUMB ABDUCT/ADDUCT for spread); the module
constants below are fallbacks (measured on the left-glove fixtures) used only
when calibration is skipped (--no-calibrate).

No radians, no invert_flex: this module always computes the L6-native
convention (100=open, 0=closed) directly, so it needs no whole-hand sign flag
(contrast prehensile.command.qpos_to_l6_angles). The thumb ABDUCTION channel is
the exception -- it reads through a proxy whose sign is not reliable across
gloves, so it takes a per-glove complement (``abd_invert``) supplied from
profiles.GloveProfile.

Pure numpy (no CAN/mujoco imports) -- safe to import for dry-run and tests.
"""

from __future__ import annotations

import numpy as np

from prehensile.command import L6_DRIVER_JOINTS, L6_OPEN, L6_SDK_ORDER
from prehensile.hand import Channel, HandDescriptor, Output

# Kinematic chains: wrist(0) -> mcp -> pip -> dip -> tip, one per flex channel.
# Matches fk._TIP_INDICES tip order (thumb, index, middle, ring, pinky).
_FLEX_CHAINS: dict[str, tuple[int, int, int, int, int]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
_FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
# Fingers whose flex channel uses the chord-ratio metric (everything but the
# thumb, which has its own joint-bend metric below).
_CHORD_FINGERS: tuple[str, ...] = ("index", "middle", "ring", "pinky")
# finger name -> output slot name in command.L6_SDK_ORDER
_FLEX_SLOT: dict[str, str] = {
    "thumb": "thumb_flex", "index": "index", "middle": "middle",
    "ring": "ring", "pinky": "pinky",
}

# Fallback chord-ratio bounds for the four non-thumb fingers, measured on the
# real left-glove fixture recordings (tests/fixtures/quat_{open,fist}_3s.bin):
# open decodes to r_f in ~0.95-1.0, fist to ~0.34-0.49 per finger.
R_OPEN: dict[str, float] = {
    "index": 0.95, "middle": 0.99, "ring": 0.99, "pinky": 0.98,
}
R_CLOSED: dict[str, float] = {
    "index": 0.44, "middle": 0.48, "ring": 0.35, "pinky": 0.49,
}
# Thumb-flexion fallback bounds (total MCP+IP bend, degrees). open = small bend
# (thumb extended), closed = large bend (thumb curled). Measured on the real
# left-glove fixtures (quat_open/fist_3s.bin project to ~43/119 deg); prefer the
# live THUMB EXTENDED/CURL calibration gestures.
THUMB_FLEX_OPEN_DEG: float = 43.4
THUMB_FLEX_CLOSED_DEG: float = 119.0
# Thumb-abduction fallback bounds (out-of-plane metacarpal elevation, SIGNED
# degrees). Rough defaults only -- the sign and scale depend on hand side, and a
# fist can't calibrate abduction (it wraps the thumb), so these are not
# fixture-derived. Always prefer the live THUMB ABDUCT/ADDUCT calibration.
ABD_SPREAD: float = 30.0
ABD_TUCK: float = -5.0

_EPS = 1e-9
_MID = L6_OPEN / 2.0  # pivot for the response-gain (amplify/soften around center)
# Channel indices for the thumb<-index pinch and the MRP grasp group used to be
# module-level constants resolved off the literal L6_SDK_ORDER position. Now
# that a hand's channel order/count/names are no longer fixed (see
# ``prehensile.hand.HandDescriptor``), those indices are resolved PER INSTANCE
# in ``CurlMapper.__init__`` instead (``self._i_pinch_driver``/
# ``self._i_pinch_driven``/``self._i_group``) -- by ROLE, through
# ``HandDescriptor.index_of_role``, rather than by name or position. The
# never-hardcode-a-slot-position discipline is unchanged; only the lookup key
# (role, not literal position) and the lifetime (per instance, not per module)
# have moved.

# Default static "home gesture" (a thumbs-up): the pose CurlMapper.home_gesture()
# emits while the arms are still homing, before glove teleop takes over. Given in
# PHYSICAL-openness terms (100 = open, 0 = closed on either hand) -- NOT SDK
# space -- exactly like the coupled-thumb quantity in __call__ below; overridable
# per-channel via tuning's "home_gesture" key (see CurlMapper.__init__).
HOME_GESTURE_OPENNESS: dict[str, float] = {
    "thumb_flex": 100.0,   # thumb extended
    "thumb_abd": 100.0,    # thumb spread away from the fingers
    "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0,   # fist
}

# The default hand: an IDENTITY role map off L6_SDK_ORDER (channel name == role
# name), which is why nothing changes for a caller that never passes `hand=` --
# every role literal elsewhere in this module resolves through
# HandDescriptor.index_of_role to the exact integer L6_SDK_ORDER.index(...)
# resolved to before this module had a HandDescriptor at all. Each channel's
# `home` is sourced from HOME_GESTURE_OPENNESS above, so the shipped home-gesture
# numbers do not change either.
#
# Built HERE rather than in prehensile.command (where L6_SDK_ORDER/
# L6_DRIVER_JOINTS themselves live) because it needs HOME_GESTURE_OPENNESS,
# which is a curl-map concept -- and prehensile.command must not import
# prehensile.curlmap (curlmap already imports command), so building it on this
# side is what avoids the cycle.
L6_HAND: HandDescriptor = HandDescriptor(
    name="realhand_l6",
    channels=tuple(
        Channel(name=slot, role=slot, home=HOME_GESTURE_OPENNESS[slot])
        for slot in L6_SDK_ORDER
    ),
    # pinch/group are left at HandDescriptor's own defaults -- driver index,
    # driven thumb_flex, group middle+ring+pinky -- which is exactly the
    # shipped L6 wiring.
    output=Output(units="percent", open=L6_OPEN, closed=0.0),
    default_tuning="prehensile/configs/curl_tuning.yml",
    driver_joints=dict(L6_DRIVER_JOINTS),
)


def _chain_length(kp: np.ndarray, chain: tuple[int, ...]) -> float:
    """Sum of the 4 consecutive segment lengths of ``chain`` (wrist->...->tip)."""
    return float(
        sum(np.linalg.norm(kp[chain[i + 1]] - kp[chain[i]]) for i in range(len(chain) - 1))
    )


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors (0 if either is ~zero)."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    cos_t = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _lerp_percent(value: float, lo: float, hi: float) -> float:
    """clip((value - lo) / (hi - lo), 0, 1) * L6_OPEN; robust to hi == lo.

    ``lo``/``hi`` need not be ordered: passing lo > hi simply reverses the ramp
    (used by the thumb-flex channel, where 'open' is the SMALLER bend angle)."""
    denom = hi - lo
    if abs(denom) < _EPS:
        denom = _EPS if denom >= 0 else -_EPS
    return float(np.clip((value - lo) / denom, 0.0, 1.0) * L6_OPEN)


def _thumb_flex_bend_deg(kp: np.ndarray) -> float:
    """Total thumb joint-bend angle (MCP bend + IP bend), degrees.

    ~0 for a straight thumb, growing monotonically as it curls. Uses the thumb
    chain landmarks kp[1..4] (cmc, mcp, ip, tip). Unlike the chord ratio this
    has a large dynamic range for the (short, opposing) thumb and does not fold
    back at high flexion."""
    mcp = _vec_angle_deg(kp[2] - kp[1], kp[3] - kp[2])
    ip = _vec_angle_deg(kp[3] - kp[2], kp[4] - kp[3])
    return mcp + ip


def _palm_normal(kp: np.ndarray) -> np.ndarray:
    """Unit normal of the palm plane spanned by wrist->index-mcp (kp5-kp0) and
    wrist->pinky-mcp (kp17-kp0).

    The sign is arbitrary (the normal is used only to remove an out-of-plane
    component), so the result is hand-side independent. Returns a zero vector if
    the span is degenerate."""
    n = np.cross(kp[5] - kp[0], kp[17] - kp[0])
    norm = float(np.linalg.norm(n))
    return n / norm if norm > _EPS else n


def _thumb_abd_angle_deg(kp: np.ndarray) -> float:
    """Thumb palmar-abduction proxy: elevation of the thumb metacarpal
    (kp1->kp2) OUT of the palm plane, in signed degrees [-90, 90].

    = arcsin( unit(kp2-kp1) . palm_normal ). Palmar abduction lifts the thumb
    perpendicular to the palm (out of plane); flexion curls it across the palm
    (in plane). Measuring the out-of-plane component is far less
    flexion-confounded than the old in-plane angle (on real UDCap-left data the
    correlation with flexion dropped ~0.88 -> ~0.69 with ~2.5x more
    flexion-independent signal). The sign depends on the palm normal's arbitrary
    orientation and on hand side; the calibration spread/tuck bounds and
    ``abd_invert`` absorb it."""
    n = _palm_normal(kp)
    meta = kp[2] - kp[1]
    nm = float(np.linalg.norm(meta))
    if nm < _EPS or float(np.linalg.norm(n)) < _EPS:
        return 0.0
    return float(np.degrees(np.arcsin(np.clip(float(np.dot(meta, n)) / nm, -1.0, 1.0))))


class CurlMapper:
    """Stateful (21,3) MediaPipe keypoints -> per-channel L6-openness angles
    [0,100], EMA-smoothed, in the ORDER of ``hand`` (``L6_HAND`` by default).

    ``side`` selects the hand. The curl/bend/spread metrics themselves are
    side-independent, and ``side`` no longer flips the thumb by itself -- it is
    accepted/stored for API symmetry only. A mirrored thumb (the L6 thumbs can
    be mounted/oriented either way on both hands) is flipped via the tuning
    ``flip`` key instead (see the per-side ``left``/``right`` sections in
    configs/curl_tuning.yml).

    ``hand`` is an optional ``prehensile.hand.HandDescriptor`` (keyword-only,
    defaults to ``L6_HAND`` -- an identity role map off ``L6_SDK_ORDER``, so
    every existing caller that never passes it is unaffected byte-for-byte).
    It is what lets this class drive a hand whose channel names, order, and
    count differ from the L6's: every place below that used to say "the
    thumb_flex slot" or "L6_SDK_ORDER position N" now means "whichever channel
    this hand's descriptor gives that ROLE" (see ``prehensile.hand``'s module
    docstring for what a role is and why an unmapped channel cannot exist).
    The six metrics this class computes from a keypoint frame are themselves
    always the same six roles, on every hand; only which output channel
    receives each one -- and that hand's grasp-group/pinch wiring -- changes.

    ``r_open``/``r_closed`` are ``{finger: float}`` dicts over the four chord
    fingers ``{"index","middle","ring","pinky"}`` (default: module ``R_OPEN``/
    ``R_CLOSED``). ``thumb_flex_bounds`` is a ``{"open": float, "closed": float}``
    dict of MCP+IP bend degrees (default: module ``THUMB_FLEX_OPEN_DEG``/
    ``THUMB_FLEX_CLOSED_DEG``). ``abd_bounds`` is a ``{"spread": float, "tuck":
    float}`` dict of palm-plane-projected degrees (default: module
    ``ABD_SPREAD``/``ABD_TUCK``).

    Response tuning (all default to no-op):
      ``flex_gain`` scales the five flex channels' swing around the 50 midpoint
        (>1 amplifies the open/close motion, <1 softens it).
      ``abd_gain`` does the same for the thumb_abd channel (<1 makes abduction
        change more gently / less abruptly).
      ``thumb_flex_pivot`` sets the gain's fixed point for thumb_flex only
        (``None`` -> the 50 midpoint like the other channels). Only matters when
        ``flex_gain != 1``; raising it (e.g. 70) pushes sub-pivot (curled)
        values lower for a given gain.
      ``alpha`` is the fallback EMA smoothing factor for every channel; a
        per-channel ``alpha`` in ``tuning`` (below) overrides it for that
        channel only (e.g. a slower thumb_abd to de-jitter abduction).
      ``tuning`` is an optional ``{channel: {"gain"|"pivot"|"alpha"|"park":
        float, "flip": bool}}`` dict (typically loaded and per-side-resolved
        from configs/curl_tuning.yml by ``prehensile.tuning.resolve_tuning``)
        that overrides the scalar-derived per-channel gain/pivot/alpha above
        and/or marks a channel for output-flip (``100 - x``, applied last --
        e.g. to mirror a thumb mounted backwards on one hand). ``park`` seeds
        that channel's parked value (see ``self.locked``/``set_park`` below).

    Park-lock (grasp thumb posture):
      ``self.locked`` is a public runtime flag (default ``False``). While
      ``True``, every channel with a non-``None`` parked value is forced to
      that value as the FINAL output -- a literal 0-100 L6 command applied
      AFTER the per-channel ``flip`` (and bypassing gain/pivot), so every hand
      is commanded exactly the same value. park is a fixed output target, so
      the ``flip`` -- which only corrects the tracking metric's per-side mirror
      -- must NOT touch it (else a flipped channel would park to ``100 - park``
      on that hand). The EMA keeps tracking underneath, so unlocking resumes
      without a jump. A channel's parked value comes from ``tuning``'s ``park``
      key and/or ``set_park`` (which also works live, e.g. from a discovery
      tool). ``locked=False`` leaves every channel's normal mapped behaviour
      byte-for-byte unchanged.

      ``self.last_unparked`` is the 6-length output as of the same call, taken
      after clip+flip but BEFORE the park block above overwrites any channels
      -- i.e. the value each channel would have produced without the park
      override, updated every call regardless of ``locked``. It is always an
      independent copy (never an alias of the returned list), since the park
      block mutates the returned list in place and aliasing would silently
      corrupt this snapshot too. This is what a UI (``prehensile.teleop``'s
      console readout) should display for a parked channel while ``locked`` is
      True, so the operator keeps seeing the live tracked value underneath the
      frozen hardware command; it is ``None`` exactly when ``__call__`` itself
      returned ``None`` and has not yet produced a first frame (and, like
      ``self._last``, is left unchanged by a call that returns ``None``).

    Thumb<-index coupling (proportional pinch):
      ``self.couple_thumb_index`` is a second public runtime flag (default
      ``True``, set from the constructor kwarg of the same name). While it AND
      ``self.locked`` are both ``True``, ``thumb_flex`` stops using its own
      tracked metric and is driven linearly off the index channel, rescaled onto
      ``[couple_low, 100]`` -- index fully open gives a fully open thumb, index
      fully closed gives ``couple_low``. ``couple_low`` comes from ``tuning``'s
      ``thumb_flex`` entry and defaults to ``0.0`` (full travel) when unset,
      including when ``tuning`` is ``None``.

      A ``couple_low`` on the ``index`` entry is a second, separate floor: it
      clamps what the DRIVING finger itself is commanded, so the index stops short
      of fully closed (see ``couple_index_floor``). Unset/0 means no clamp. The
      index is then normalized within its resulting WORKING window
      ``[index_low, 100]`` rather than the raw ``[0, 100]``, so the thumb still
      spends its full ``[couple_low, 100]`` range over the travel the finger
      actually has, and ``couple_low`` remains the thumb's true floor for any
      index floor. (Normalizing over the raw range instead would let the index
      floor silently raise the thumb's floor to
      ``couple_low + index_low/100 * (100 - couple_low)``.) Both channels reach
      their own floor together. ``index_low == 100`` is a degenerate window -- the
      index cannot move -- and pins both channels fully open.
      Note this makes ``thumb_flex``'s own ``gain``/``pivot`` irrelevant to the
      commanded thumb while coupled -- they still shape the tracked value in
      ``last_unparked`` and the EMA state unlocking resumes from, but the coupled
      command is a function of the index alone.

      This differs from ``park`` in one important way: a park is a literal SDK
      command, so the flip must not touch it, whereas a coupled value is a
      TRACKED quantity. It is therefore computed from the index's *physical*
      openness (pre-flip) and then converted into ``thumb_flex``'s own SDK space
      by applying ``thumb_flex``'s flip -- otherwise a mirrored thumb would close
      as the index opened. It is applied after the park override (so the narrower
      opt-in wins if a channel carries both), after the ``last_unparked``
      snapshot, and after the EMA (so ``thumb_flex``'s filter keeps tracking the
      real thumb underneath and unlocking resumes without a jump).

      ``self.parked_channels`` is a read-only ``tuple[str, ...]`` property
      listing this hand's channel names that currently have a non-``None``
      parked value (``()`` if nothing is parked), in the hand's own channel
      order. It is computed live from the parked-value table, so it reflects
      ``tuning``-seeded parks as well as any live ``set_park`` calls,
      including clearing a slot back out with ``set_park(slot, None)``.

    MRP group (median grasp of middle/ring/pinky):
      Unconditional while ``self.locked`` is ``True`` -- there is no separate
      flag for it, unlike ``couple_thumb_index``. It is simply part of what the
      grasp lock means: ``middle``, ``ring`` and ``pinky`` are commanded as ONE
      group, each set to the MEDIAN of the three fingers' tracked (physical
      openness) values.

      Median, specifically, and not mean or min/max:
        - Median is one of the three real fingers' own values, so the group
          reaches what the consensus finger reaches. A mean would always sit
          less committed than the operator's most-committed finger -- a
          looser grip than intended -- and would launder the ring finger's
          outlier calibration bound (``R_CLOSED`` 0.35 vs 0.48/0.49 for
          middle/pinky) into all three channels, whereas the median rejects
          exactly that kind of single-channel skew.
        - Median is the only 3-input selector that survives one bad channel in
          BOTH directions: ``min`` is a single-fault path to closure (one low
          glitch closes all three onto whatever is in the hand), ``max`` a
          single-fault path to release.
        - Safety property: the median of three values always lies within
          ``[min, max]`` of those same three values, so this can never command
          any of the three channels to a pose it could not already reach on
          its own -- the group only ever picks among poses already in play.

      Like the coupled thumb (and unlike ``park``), the group value is a
      TRACKED quantity, not a literal SDK target: it is computed from ``pre``
      (each finger's physical openness, pre-flip) and then written into each
      of the three channels through THAT channel's own ``flip`` -- not a
      single shared flip -- so a mirrored MRP channel would still close as the
      others close. (No MRP channel is flipped in the shipped config, so this
      is a no-op there today; it is still implemented, since nothing else
      about the coupling makes that assumption.) It is computed after the
      thumb<-index coupling block, but the two are independent -- the group
      reads only ``pre`` (which nothing mutates) and its channel set
      (``middle``/``ring``/``pinky``) is disjoint from the thumb block's
      (``thumb_flex``/``index``), so the ordering between the two blocks is
      cosmetic; appending the group after the thumb block keeps that
      hardware-validated block's diff untouched. Like the coupled thumb, it is
      applied after the park loop (the narrower opt-in wins, so a park on an
      MRP channel is overwritten by the group -- unreachable in the shipped
      config, but implemented the same way for consistency), after the
      ``last_unparked`` snapshot (so a UI can still show the operator's real
      fingers underneath the group), and after the EMA (so each channel's
      filter keeps tracking the real finger underneath and unlocking resumes
      without a jump).

    Static home gesture (thumbs-up while the arms home):
      ``home_gesture()`` returns a fixed 6-channel pose -- default
      ``HOME_GESTURE_OPENNESS`` (a thumbs-up: thumb extended and spread, the
      other four fingers curled into a fist) -- for the ROS node to command
      while the arms are still homing, before glove teleop takes over. Like
      the coupled thumb (and unlike ``park``), the gesture is a
      physical/tracked quantity, defined in PHYSICAL-openness terms, so it is
      converted into this hand's SDK space by applying each channel's
      ``flip`` on the way out -- otherwise a mirrored ``thumb_abd`` would
      point the thumb the wrong way on that hand instead of mirroring the
      other hand's pose. ``tuning``'s optional per-channel ``home_gesture``
      key overrides the default for that channel (clamped to ``[0,
      L6_OPEN]`` at ingest, like ``couple_low``). The method is pure: it does
      not read or mutate ``self._last``, ``self.locked``, ``self._parks``, or
      the thumb<-index coupling, so calling it never disturbs the EMA.
    """

    def __init__(
        self,
        side: str = "left",
        alpha: float = 0.4,
        r_open: dict[str, float] | None = None,
        r_closed: dict[str, float] | None = None,
        abd_bounds: dict[str, float] | None = None,
        abd_invert: bool = False,
        thumb_flex_bounds: dict[str, float] | None = None,
        flex_gain: float = 1.0,
        abd_gain: float = 1.0,
        thumb_flex_pivot: float | None = None,
        tuning: dict[str, dict[str, float | bool]] | None = None,
        couple_thumb_index: bool = True,
        *,
        hand: HandDescriptor | None = None,
    ) -> None:
        # The hand descriptor this mapper drives. Defaults to L6_HAND (an
        # identity role map off L6_SDK_ORDER), so every existing caller that
        # never passes `hand=` -- in particular hand_teleop_node's fixed
        # 5-kwarg construction -- is unaffected. `hand` is keyword-only and
        # last, per the same discipline as every other constructor addition
        # here: existing positional callers cannot be broken by it.
        self.hand: HandDescriptor = L6_HAND if hand is None else hand
        _order = self.hand.order
        _roles = self.hand.roles
        # Grasp-group channel indices, resolved by ROLE (not by name/position)
        # through the descriptor -- collects EVERY channel whose role is in
        # self.hand.group, so a role claimed by more than one channel puts all
        # of them in the group (see prehensile.hand's module docstring).
        self._i_group: tuple[int, ...] = tuple(
            i for i, r in enumerate(_roles) if r in self.hand.group
        )
        _pinch = self.hand.pinch
        # Pinch driver/driven channel indices, resolved by role; None on a hand
        # with no pinch configured (or, defensively, one whose pinch role
        # somehow resolves to nothing -- HandDescriptor.__post_init__ already
        # guarantees this cannot happen for a validated descriptor).
        self._i_pinch_driver: int | None = (
            None if _pinch is None else self.hand.index_of_role(_pinch.driver)
        )
        self._i_pinch_driven: int | None = (
            None if _pinch is None else self.hand.index_of_role(_pinch.driven)
        )
        self.side = side
        self.alpha = alpha
        self.r_open = dict(R_OPEN if r_open is None else r_open)
        self.r_closed = dict(R_CLOSED if r_closed is None else r_closed)
        bounds = {"spread": ABD_SPREAD, "tuck": ABD_TUCK} if abd_bounds is None else abd_bounds
        self.abd_spread = float(bounds["spread"])
        self.abd_tuck = float(bounds["tuck"])
        tfb = ({"open": THUMB_FLEX_OPEN_DEG, "closed": THUMB_FLEX_CLOSED_DEG}
               if thumb_flex_bounds is None else thumb_flex_bounds)
        self.thumb_flex_open = float(tfb["open"])
        self.thumb_flex_closed = float(tfb["closed"])
        # The base-segment abduction proxy is flexion-confounded, so its sign is
        # not reliable across gloves (reversed on Wuji). abd_invert flips the
        # thumb_abd channel to the hardware-native sense when that happens.
        self.abd_invert = bool(abd_invert)
        self.flex_gain = float(flex_gain)
        self.abd_gain = float(abd_gain)
        # Gain pivot (the value the gain leaves unchanged) for thumb_flex only;
        # every other channel pivots at the 50 midpoint. Raising it (e.g. 70)
        # pushes sub-pivot (curled) values lower for a given flex_gain.
        self.thumb_flex_pivot = _MID if thumb_flex_pivot is None else float(thumb_flex_pivot)
        # Per-channel response gain, EMA alpha and gain pivot, aligned to this
        # hand's own channel order (thumb_abd ROLE gets its own gain; thumb_flex
        # ROLE its own pivot; every channel starts at the fallback `alpha`,
        # overridable per-channel below via tuning's "alpha" key).
        self._gains = [self.abd_gain if role == "thumb_abd" else self.flex_gain
                       for role in _roles]
        self._alphas = [self.alpha for _ in _roles]
        self._pivots = [self.thumb_flex_pivot if role == "thumb_flex" else _MID
                        for role in _roles]
        # Per-channel gain/pivot/flip overrides (e.g. from configs/curl_tuning.yml
        # via prehensile.tuning); gain/pivot take precedence over the
        # scalar-derived values above. A truthy "flip" for a channel marks its
        # slot for the output complement below (e.g. a thumb mounted backwards
        # on one hand); this is the ONLY source of flips -- ``side`` itself no
        # longer implies one.
        self._flip_slots: list[int] = []
        # Per-channel parked value (native 0-100 convention), forced onto the
        # channel -- bypassing gain/pivot -- while self.locked is True. None
        # means that channel has no park set (see set_park() to set/clear live).
        self._parks: list[float | None] = [None] * len(_order)
        # Lower bound of the thumb<-index coupling, in physical-openness terms.
        # 0.0 (the default when unset, incl. tuning=None from --no-tune or a
        # missing config) lets the thumb follow the index all the way closed.
        self._couple_low: float = 0.0
        # Floor on the INDEX's own command while coupled: the driving finger stops
        # short instead of closing all the way, and the thumb maps off the clamped
        # value so both saturate together. 0.0 (default) means no clamp.
        self._couple_index_low: float = 0.0
        # Static home-gesture pose, in PHYSICAL-openness terms (see
        # prehensile.hand.Channel.home / home_gesture()), aligned to this
        # hand's own channel order. Sourced from the descriptor's per-channel
        # `home` (for L6_HAND, HOME_GESTURE_OPENNESS's values verbatim -- see
        # L6_HAND above); clamped like every other physical-openness value
        # ingested here. A tuning "home_gesture" entry below overrides
        # individual channels.
        self._home_gesture_pre: list[float] = [
            float(np.clip(c.home, 0.0, L6_OPEN)) for c in self.hand.channels
        ]
        # The pinch's driver/driven ROLES (not channel names), used below to
        # match a tuning "couple_low" entry to the right side of the coupling
        # regardless of what this hand happens to call its index/thumb
        # channels. None on a hand with no pinch configured, in which case a
        # "couple_low" entry (on any channel) is simply never claimed here --
        # matching couple_thumb_index/self._i_pinch_driven being inert too.
        _pinch_driven_role = None if _pinch is None else _pinch.driven
        _pinch_driver_role = None if _pinch is None else _pinch.driver
        if tuning:
            for i, (slot, role) in enumerate(zip(_order, _roles)):
                ch = tuning.get(slot) or {}
                if "gain" in ch:
                    self._gains[i] = float(ch["gain"])
                if "pivot" in ch:
                    self._pivots[i] = float(ch["pivot"])
                if "alpha" in ch:
                    self._alphas[i] = float(ch["alpha"])
                if ch.get("flip"):
                    self._flip_slots.append(i)
                if "park" in ch:
                    self._parks[i] = float(ch["park"])
                # couple_low means different things on the coupling's two
                # roles; prehensile.tuning rejects it anywhere else, but a
                # caller can hand us a raw dict, so match on the ROLE (this
                # hand's own pinch wiring), not a hardcoded channel name.
                if "couple_low" in ch:
                    # Clamped on the way in: an out-of-range floor would otherwise
                    # distort the window normalization in __call__, not just the
                    # output value.
                    cl = float(np.clip(float(ch["couple_low"]), 0.0, L6_OPEN))
                    if role == _pinch_driven_role:
                        self._couple_low = cl
                    elif role == _pinch_driver_role:
                        self._couple_index_low = cl
                if "home_gesture" in ch:
                    # Clamped on the way in, like couple_low above -- an
                    # out-of-range pose value should never escape to home_gesture().
                    self._home_gesture_pre[i] = float(
                        np.clip(float(ch["home_gesture"]), 0.0, L6_OPEN)
                    )
        # Runtime thumb<-index coupling flag (public, like self.locked): while
        # this AND self.locked are both True, thumb_flex is driven off the index
        # instead of its own metric (see the class docstring).
        self.couple_thumb_index: bool = bool(couple_thumb_index)
        # Runtime park-lock flag: while True, __call__ forces every slot with
        # a non-None parked value to that value (see the class docstring).
        self.locked: bool = False
        self._last: list[float] | None = None
        self.last_unparked: list[float] | None = None

    def set_park(self, slot: str, value: float | None) -> None:
        """Set (or clear, with None) the parked value for one channel, applied when self.locked."""
        self._parks[self.hand.order.index(slot)] = None if value is None else float(value)

    @property
    def couple_index_floor(self) -> float | None:
        """The floor applied to the index's own command while coupled, or ``None``
        when unset. Surfaced so a UI can mark the index as not-tracking (its
        displayed value is the tracked one, not the clamped command)."""
        return self._couple_index_low if self._couple_index_low > 0.0 else None

    @property
    def parked_channels(self) -> tuple[str, ...]:
        """This hand's channel names that currently have a non-None parked
        value, in the hand's own channel order (``()`` if nothing is parked).

        Computed live off ``self._parks`` on every access, so it always
        reflects the current state -- whether seeded from ``tuning`` at
        construction or changed since via ``set_park`` (including clearing a
        slot with ``set_park(slot, None)``)."""
        return tuple(slot for slot, pv in zip(self.hand.order, self._parks) if pv is not None)

    def home_gesture(self) -> list[float]:
        """The static home-gesture pose as a per-hand SDK command: one float per
        channel, in this hand's own channel order, ready to send straight to
        this hand's ``set_angles``.

        Default ``HOME_GESTURE_OPENNESS`` (a thumbs-up), overridable per-channel
        via ``tuning``'s ``home_gesture`` key -- see ``self._home_gesture_pre``,
        resolved once at construction.

        The gesture is a physical/tracked quantity (like the coupled thumb),
        NOT a literal SDK target like ``park`` -- so unlike ``park`` it passes
        THROUGH each channel's ``flip`` on the way out: ``self._home_gesture_pre``
        is stored in physical-openness terms shared by both hands, and
        ``L6_OPEN - v`` is applied for every ``i in self._flip_slots`` to convert
        it into this hand's SDK space. Skipping the flip here would leave a
        mirrored ``thumb_abd`` pointing the thumb the wrong way on that hand
        instead of reproducing the same physical thumbs-up as the other hand.

        Pure: reads only ``self._home_gesture_pre``/``self._flip_slots`` (both
        fixed at construction), and never touches ``self._last``,
        ``self.locked``, ``self._parks``, or the thumb<-index coupling -- so
        calling it never disturbs the EMA.
        """
        out = list(self._home_gesture_pre)
        for i in self._flip_slots:
            out[i] = L6_OPEN - out[i]
        return out

    def _flex_percent(self, kp: np.ndarray, finger: str) -> float:
        chain = _FLEX_CHAINS[finger]
        r_f = float(np.linalg.norm(kp[chain[-1]] - kp[chain[0]])) / _chain_length(kp, chain)
        return _lerp_percent(r_f, self.r_closed[finger], self.r_open[finger])

    def _thumb_flex_percent(self, kp: np.ndarray) -> float:
        # open = small bend -> 100, closed = large bend -> 0 (lo=closed, hi=open).
        return _lerp_percent(_thumb_flex_bend_deg(kp), self.thumb_flex_closed, self.thumb_flex_open)

    def _thumb_abd_percent(self, kp: np.ndarray) -> float:
        pct = _lerp_percent(_thumb_abd_angle_deg(kp), self.abd_tuck, self.abd_spread)
        return (L6_OPEN - pct) if self.abd_invert else pct

    def __call__(self, kp) -> list[float] | None:
        """One (21,3) keypoint frame -> 6 EMA-smoothed L6 angles, or None.

        Returns ``None`` (without touching the EMA state) if ``kp`` is not a
        finite ``(21,3)`` array or any flex chain has ~zero length.
        """
        kp = np.asarray(kp, dtype=np.float64)
        if kp.shape != (21, 3) or not np.all(np.isfinite(kp)):
            return None
        lengths = {f: _chain_length(kp, _FLEX_CHAINS[f]) for f in _FINGERS}
        if any(length <= _EPS for length in lengths.values()):
            return None

        raw = {_FLEX_SLOT[f]: self._flex_percent(kp, f) for f in _CHORD_FINGERS}
        raw["thumb_flex"] = self._thumb_flex_percent(kp)
        raw["thumb_abd"] = self._thumb_abd_percent(kp)
        # Response gain: scale each channel's swing around its pivot (50 for all
        # but thumb_flex, which may use thumb_flex_pivot); flex_gain / abd_gain
        # both 1.0 == no change. Then clip to [0,100].
        new = [
            float(np.clip(p + (raw[role] - p) * g, 0.0, L6_OPEN))
            for role, g, p in zip(self.hand.roles, self._gains, self._pivots)
        ]

        if self._last is None:
            self._last = new  # first valid frame seeds the filter, unlagged
        else:
            # Per-channel EMA (a per-channel tuning "alpha" may override any slot).
            self._last = [a * n + (1.0 - a) * last
                          for a, n, last in zip(self._alphas, new, self._last)]
        # `pre` is every channel in PHYSICAL-openness terms (100 = physically
        # open on either hand); `out` is the per-hand SDK command space the flip
        # converts into. The coupling below needs the former, so keep both.
        pre = [float(v) for v in np.clip(self._last, 0.0, L6_OPEN)]
        out = list(pre)
        for i in self._flip_slots:  # tuning-driven output flip (100-x), e.g. a mirrored thumb
            out[i] = L6_OPEN - out[i]

        # Snapshot the pre-park output (post clip+flip) as an INDEPENDENT copy --
        # `list(out)`, not `out` itself -- so a UI can show the tracked value
        # underneath a park lock (see the class docstring). The park block just
        # below mutates `out` in place; aliasing here would let that mutation
        # silently corrupt last_unparked too.
        self.last_unparked = list(out)

        if self.locked:
            # Park-lock: force parked channels to their literal park value as the
            # FINAL output, AFTER the flip. park is a fixed L6 output target, so
            # every hand is commanded exactly `pv` -- the flip corrects the
            # tracking metric's per-side mirror, NOT a fixed target, so it must
            # not touch park (else a flipped channel parks to 100-pv on that
            # hand). The EMA above keeps tracking underneath, so unlocking
            # resumes without a jump.
            for i, pv in enumerate(self._parks):
                if pv is not None:
                    out[i] = float(np.clip(pv, 0.0, L6_OPEN))

            if self.couple_thumb_index and self._i_pinch_driven is not None:
                # thumb_flex stops using its own metric and follows the index
                # linearly, rescaled onto [couple_low, 100]: index fully open ->
                # thumb fully open, index fully closed -> thumb at couple_low.
                #
                # Unlike park, this is a TRACKED quantity, not a literal SDK
                # command -- so it is computed from the index's PHYSICAL openness
                # (`pre`, before that channel's flip) and then converted into
                # thumb_flex's own SDK space by applying thumb_flex's flip. Doing
                # it post-flip on both ends would drive a mirrored thumb closed as
                # the index opened. Clipped like every other write into `out`,
                # since couple_low is not range-checked at load.
                #
                # Applied AFTER the park loop, so on the (reachable) config where
                # thumb_flex carries both, the narrower opt-in wins. It is also
                # after the last_unparked snapshot, so a UI can still show the
                # operator's real thumb; and after the EMA, so thumb_flex's filter
                # keeps tracking underneath and unlocking resumes without a jump.
                #
                # i_drv/i_dvn are this hand's pinch driver/driven channel
                # indices (resolved by role in __init__), NOT hardcoded
                # positions -- a hand whose pinch is wired to different roles
                # (or different channel names/order) is driven identically.
                i_drv, i_dvn = self._i_pinch_driver, self._i_pinch_driven
                idx = pre[i_drv]
                index_low = self._couple_index_low
                if index_low > 0.0:
                    # Floor the DRIVING finger too: it stops short of fully closed.
                    # The floor is a statement about physical openness, so it is
                    # clamped here in `pre` space and then converted into this hand's
                    # SDK space -- same reasoning as the coupled thumb below.
                    idx = max(index_low, idx)
                    out[i_drv] = L6_OPEN - idx if i_drv in self._flip_slots else idx
                # Normalize the index within its WORKING window [index_low, 100]
                # rather than the raw [0, 100]. Without this the floor would eat
                # into the thumb's travel too -- the thumb would bottom out at
                # `low + index_low/100 * (100 - low)` instead of `low`, silently
                # making couple_low not the floor it claims to be. Remapping keeps
                # the thumb's full range over the travel the finger actually has,
                # so both channels reach their own floor together.
                span = L6_OPEN - index_low
                # span == 0 (index_low == 100) is a degenerate window: the index
                # cannot move and sits fully open, so treat the thumb as fully open
                # too -- the safe direction, and it avoids a divide-by-zero.
                u = 1.0 if span <= _EPS else float(np.clip((idx - index_low) / span, 0.0, 1.0))
                low = self._couple_low
                v = float(np.clip(low + u * (L6_OPEN - low), 0.0, L6_OPEN))
                out[i_dvn] = L6_OPEN - v if i_dvn in self._flip_slots else v

            # Grasp group (default middle/ring/pinky on L6; a hand's own
            # self.hand.group role list on any other hand): every channel whose
            # role is in the group moves as ONE group, each commanded the
            # MEDIAN of the group's tracked (physical-openness) values.
            # Unconditional -- part of what the grasp lock means, exactly like
            # the thumb pinch above, not a separate opt-in flag. See the class
            # docstring's "MRP group" section for why median (not mean/min/max);
            # that safety property (the median of N values always lies within
            # [min, max] of those values) holds for any N, including a role
            # claimed by more than one channel.
            #
            # Like the coupled thumb, this is a TRACKED quantity, not a literal
            # SDK target -- computed from `pre` (physical openness, pre-flip)
            # and written out through EACH channel's OWN flip (not one shared
            # flip), so a mirrored group channel would still close as the others
            # close. (No group channel is flipped in the shipped L6 config, so
            # this is a no-op there today; implemented correctly anyway.) It
            # reads only `pre`, which nothing mutates, and (for the shipped L6
            # config) its channel set is disjoint from the thumb block's above,
            # so the two are independent there and the ordering between them is
            # cosmetic -- appending here keeps the hardware-validated thumb
            # block's diff untouched. It is applied after the park loop
            # (narrower opt-in wins, so a park on a group channel is overwritten
            # -- unreachable in the shipped config, but implemented the same way
            # for consistency), after the last_unparked snapshot (so a UI can
            # still show the operator's real fingers underneath the group), and
            # after the EMA (so each channel's filter keeps tracking the real
            # finger underneath and unlocking resumes without a jump).
            #
            # Guarded (not just an empty-loop no-op): np.median([]) is nan (and
            # emits a RuntimeWarning) BEFORE the write loop below would skip, so
            # a hand with an empty grasp group must not reach the median call.
            if self._i_group:
                group = float(np.median([pre[i] for i in self._i_group]))
                for i in self._i_group:
                    out[i] = L6_OPEN - group if i in self._flip_slots else group
        return out
