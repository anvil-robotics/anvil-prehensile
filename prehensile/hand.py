"""HandDescriptor: what CurlMapper needs to know about a hand that is not the L6.

Phase 4a of the upstream migration. ``prehensile.curlmap`` computes exactly six
role-shaped metrics per frame (``ROLES`` below) from the (21,3) MediaPipe
keypoint frame -- four chord-ratio finger curls plus the thumb's two decoupled
metrics -- regardless of which physical hand receives them. A ``HandDescriptor``
is the seam that says which OUTPUT CHANNEL (by name, order, and count) carries
which of those six roles, so a hand with different channel names/order/count
than the L6 can be driven without touching ``curlmap.py`` at all.

Design choices worth being explicit about:

* ``role`` lives ON ``Channel``, never in a parallel ``{channel: role}`` map.
  There is therefore no syntax for a channel with no role at all, and no
  ``role: null`` -- the two-lists-disagree bug class (a channel present in one
  list, silently absent from the other) is deleted structurally rather than
  guarded against in both directions. An unmapped channel would otherwise be
  commanded ``output.closed`` every frame with no error -- on a hand that
  closes at 0 that is a silently-clamping-shut channel.

* ``home`` also lives on ``Channel`` and has NO usable default: a channel that
  omits it fails to load (see ``__post_init__``). It is the one fact here that
  moves real hardware while nobody is watching a console (it is what the
  arms-homing sequence commands before glove teleop takes over), so unlike
  every tuning knob -- which is optional and falls back to a documented
  default -- a missing ``home`` is a load error, never a silent 0 or "off".

* A ROLE may be claimed by more than one channel (e.g. two tendons/linkages
  both driven off the same tracked curl, at possibly different per-channel
  gain/tuning). That is intentional and unrestricted: ``curlmap.CurlMapper``
  reads every channel's role independently, so replaying the same role onto
  several channels is just as well-defined as claiming it once. What IS
  restricted is a ``pinch``/``group`` role that no channel carries at all
  (nothing would move) and an unknown role (a typo silently doing nothing).

* There is deliberately no per-channel LIMITS concept here. Every clip in
  ``curlmap.py`` is the fixed ``[0, L6_OPEN]`` internal range, a math-domain
  guard, not a channel limit; the two places a real per-channel limit already
  lives are the URDF's radian limits (read by ``prehensile.viz``) and tuning's
  ``couple_low`` (a floor). Adding a third home for the same concept here would
  just be a new place for the same fact to disagree with itself.

* ``group`` is DATA (which roles get grouped) but the REDUCER is not: the
  median is hardcoded in ``curlmap.py``, not a ``reduce:`` knob here. The
  safety property specific to median -- it always lies within
  ``[min, max]`` of its inputs, so a group can never command a pose none of
  its members could already reach -- does not hold for a caller-chosen
  reducer (``mean`` would launder the least-representative member's bound into
  every channel in the group; ``min``/``max`` are single-fault paths to
  closure/release respectively). This is a policy, not a per-hand
  configuration, so it stays code.

* ``output`` records driver-facing facts -- units, the open/closed convention,
  and (as a lazily-resolved ``"module:attr"`` string, never an import) where to
  find a driver -- that ``curlmap.py`` is NOT allowed to read: the mapper's
  internal space is a fixed dimensionless 0-100 "openness percent" on every
  hand, never the hand's own convention/range. Settled here, in the file
  format and this dataclass, but genuinely unconsumed until Phase 4b, so the
  format does not need to change again when a driver shows up.

Pure stdlib (no numpy, no yaml): importable by the ROS node's lean venv and by
``curlmap.py`` without pulling in anything beyond what each already needs.
YAML parsing lives in ``prehensile.hand_loader``, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The only six metrics curlmap.py ever computes from a keypoint frame. A
# channel's role must be one of these; nothing else is meaningful to the
# mapper's arithmetic.
ROLES: frozenset[str] = frozenset(
    {"thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"}
)


@dataclass(frozen=True)
class Channel:
    """One physical output slot: its SDK name, the role it is driven as, and
    its physical-openness value (0-100, either hand) during the static home
    gesture.

    ``home`` has no default on purpose -- see the module docstring and
    ``HandDescriptor.__post_init__``, which raises if it is ``None``.
    """

    name: str
    role: str
    home: float | None = None


@dataclass(frozen=True)
class Pinch:
    """Thumb<-index proportional-pinch wiring: which role drives, which is driven.

    Both are ROLE names (resolved to a channel index via
    ``HandDescriptor.index_of_role``), not channel names, so retuning which
    physical channel plays a role never touches this.
    """

    driver: str = "index"
    driven: str = "thumb_flex"


@dataclass(frozen=True)
class Output:
    """Driver-facing facts ``curlmap.py`` never reads (see the module docstring).

    ``driver`` is an optional ``"module:attr"`` reference, resolved LAZILY (in
    Phase 4b, not here) so that reading a descriptor never forces an import --
    in particular never forces a CAN/SDK import just because a descriptor was
    loaded.
    """

    units: str = "percent"
    open: float = 100.0
    closed: float = 0.0
    driver: str | None = None


@dataclass(frozen=True)
class HandDescriptor:
    """Everything ``CurlMapper`` needs to drive a hand that is not the L6.

    ``channels`` is the ordered, authoritative channel list: ``order`` and
    ``roles`` are both derived from it, in the same order, so they can never
    drift apart from each other or from ``channels`` itself.
    """

    name: str
    channels: tuple[Channel, ...]
    pinch: Pinch | None = field(default_factory=Pinch)
    group: tuple[str, ...] = ("middle", "ring", "pinky")
    output: Output = field(default_factory=Output)
    default_tuning: str | None = None
    driver_joints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError(f"hand {self.name!r}: needs at least one channel")

        names = [c.name for c in self.channels]
        dup_names = {n for n in names if names.count(n) > 1}
        if dup_names:
            raise ValueError(f"hand {self.name!r}: duplicate channel name(s) {sorted(dup_names)}")

        for c in self.channels:
            if not c.role:
                raise ValueError(
                    f"hand {self.name!r}: channel {c.name!r} has no role; every channel "
                    f"must name a role ({sorted(ROLES)}) -- an unmapped channel would be "
                    f"silently commanded {self.output.closed!r} (this hand's closed value) "
                    "every frame"
                )
            if c.role not in ROLES:
                raise ValueError(
                    f"hand {self.name!r}: channel {c.name!r} has unknown role {c.role!r}; "
                    f"valid roles: {sorted(ROLES)}"
                )
            if c.home is None:
                raise ValueError(
                    f"hand {self.name!r}: channel {c.name!r} has no 'home' value; every "
                    "channel must set one (its physical-openness value, 0-100 on either "
                    "hand, during the static home gesture) -- there is no default, since "
                    "the wrong value moves real hardware while the arms are still homing "
                    "and nobody is watching a console"
                )

        roles_present = {c.role for c in self.channels}

        if self.pinch is not None:
            for what, role in (("driver", self.pinch.driver), ("driven", self.pinch.driven)):
                if role not in ROLES:
                    raise ValueError(f"hand {self.name!r}: pinch {what} role {role!r} unknown")
                if role not in roles_present:
                    raise ValueError(
                        f"hand {self.name!r}: pinch {what} names role {role!r}, which no "
                        "channel carries"
                    )

        for role in self.group:
            if role not in ROLES:
                raise ValueError(f"hand {self.name!r}: grasp group role {role!r} unknown")
            if role not in roles_present:
                raise ValueError(
                    f"hand {self.name!r}: grasp group names role {role!r}, which no "
                    "channel carries"
                )

    @property
    def order(self) -> tuple[str, ...]:
        """Channel names, in the hand's own SDK slot order."""
        return tuple(c.name for c in self.channels)

    @property
    def roles(self) -> tuple[str, ...]:
        """Each channel's role, aligned index-for-index with ``order``."""
        return tuple(c.role for c in self.channels)

    def index_of_role(self, role: str) -> int | None:
        """The first channel index carrying ``role``, or ``None`` if no channel does.

        "First" only matters when a role is claimed by more than one channel
        (see the module docstring); ``pinch`` wants exactly one authoritative
        channel per role, while ``group`` membership (see ``curlmap.py``)
        instead collects every matching index.
        """
        for i, c in enumerate(self.channels):
            if c.role == role:
                return i
        return None

    def channel_of_role(self, role: str) -> str | None:
        i = self.index_of_role(role)
        return None if i is None else self.channels[i].name
