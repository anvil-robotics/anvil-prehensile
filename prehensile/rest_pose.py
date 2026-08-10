"""UDCap left-hand rest-pose skeleton (the FK backbone for Branch B).

Transcribed verbatim from the vendor's "Hand Model and Data Specification"
(udexreal-docs), left-hand table. Positions are the *absolute translations* in
**centimeters** and rotations are *absolute rotations* as quaternions in
**XYZW** order, both stored exactly as printed -- no pre-transform to meters, no
axis negation. Consumers (fk.py, udcap.py) apply the model->wire transform via
``to_wire_meters``.

Frame notes (verified by the controller):
  * Units are centimeters; divide by 100 for meters.
  * The table is a right-handed model frame. The protobuf/JSON WIRE the glove
    emits is Unity left-handed and equals ``(p_node - p_wrist)`` with the x
    component negated, /100 for meters:  ``wire = ([-1,1,1] * (p - p_wrist)) / 100``.

Public surface (later tasks depend on these names/types):
  REST_POS_CM     node name -> (3,) float64 position in cm.
  REST_QUAT       node name -> (4,) float64 quaternion, XYZW.
  MEDIAPIPE_NODES model node name for each MediaPipe landmark index 0..20.
  SEGMENTS        parent->child edges of the 5 finger chains (metas included).
  to_wire_meters  model-frame position -> wrist-local, x-negated, meters.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Raw table: node -> (position cm (x, y, z), rotation quat (x, y, z, w)).
# Transcribed verbatim from the vendor left-hand specification table.
# --------------------------------------------------------------------------- #
_TABLE: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "Root": (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "LeftHand": (
        (8.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "wrist_l": (
        (8.01597874425, -0.00319243781269, -0.0625709071755),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "finger_index_meta_l": (
        (7.8602542337, 2.10413049348, 1.41612639278),
        (0.531056298718, -0.351434298812, 0.539577855624, 0.550753010824),
    ),
    "finger_index_0_l": (
        (9.11995945626, 3.73568914067, 8.50215607157),
        (0.569016145238, -0.443193991622, 0.513485228274, 0.464900669598),
    ),
    "finger_index_1_l": (
        (9.46549012355, 3.61911921911, 12.815424015),
        (0.603796191044, -0.401463936032, 0.546730915815, 0.418738789031),
    ),
    "finger_index_2_l": (
        (9.69119417809, 3.54297435512, 15.6328929296),
        (0.618308575546, -0.40796017639, 0.53829823803, 0.401868146063),
    ),
    "finger_index_l_end": (
        (9.87336494624, 3.48151602069, 17.9069342562),
        (0.569016132789, -0.443194081541, 0.513485133683, 0.464900703591),
    ),
    "finger_middle_meta_l": (
        (8.23370922171, 0.708761917427, 1.56931259483),
        (0.561749696485, -0.419737279231, 0.472987949641, 0.533423185114),
    ),
    "finger_middle_0_l": (
        (9.65288047059, 0.942903224211, 8.51040237136),
        (0.52199913876, -0.463559282171, 0.518669359027, 0.493570447914),
    ),
    "finger_middle_1_l": (
        (9.79164589318, 1.06379958705, 12.817325876),
        (0.530584650603, -0.492680618001, 0.487910867419, 0.487533304139),
    ),
    "finger_middle_2_l": (
        (9.89872809538, 1.15709343839, 16.1408848075),
        (0.521999087325, -0.463559195508, 0.518669433374, 0.493570505576),
    ),
    "finger_middle_l_end": (
        (9.98207481391, 1.22970813619, 18.7277526683),
        (0.521999087325, -0.463559195508, 0.518669433374, 0.493570505576),
    ),
    "finger_pinky_meta_l": (
        (7.76816300116, -1.90132988803, 1.45879318565),
        (0.505748072389, -0.601636397582, 0.360574230945, 0.502233766644),
    ),
    "finger_pinky_0_l": (
        (7.86898905047, -3.44992353475, 7.54976519385),
        (0.463245512768, -0.543053407924, 0.413761356463, 0.56506471394),
    ),
    "finger_pinky_1_l": (
        (8.07150440414, -3.5560677116, 10.5284357095),
        (0.436878857044, -0.580690191527, 0.374918480934, 0.575649110471),
    ),
    "finger_pinky_2_l": (
        (8.19337931858, -3.61994555155, 12.3210155097),
        (0.315477991524, -0.587963212359, 0.32905173897, 0.668204946747),
    ),
    "finger_pinky_l_end": (
        (8.3155212635, -3.68396330412, 14.1175225214),
        (0.463245493681, -0.543053338237, 0.413761417283, 0.565064752026),
    ),
    "finger_ring_meta_l": (
        (8.0673209969, -0.657705081627, 1.57219519466),
        (0.550143535319, -0.495547830323, 0.429887865514, 0.51669223092),
    ),
    "finger_ring_0_l": (
        (8.98607252697, -1.32408820287, 8.07133423232),
        (0.509447127033, -0.487278247149, 0.473873378439, 0.527700251867),
    ),
    "finger_ring_1_l": (
        (9.29262162138, -1.30940280909, 12.092763966),
        (0.517350778623, -0.517618915142, 0.442107789446, 0.518613086083),
    ),
    "finger_ring_2_l": (
        (9.5091587837, -1.2990295163, 14.9333827309),
        (0.509447157128, -0.487278282022, 0.473873344733, 0.527700220878),
    ),
    "finger_ring_l_end": (
        (9.67964551148, -1.29086257669, 17.1698936043),
        (0.509447157128, -0.487278282022, 0.473873344733, 0.527700220878),
    ),
    "finger_thumb_0_l": (
        (6.22459857725, 2.91461132653, 2.46726097912),
        (0.541194419821, 0.182029268031, 0.773035739622, 0.276386849908),
    ),
    "finger_thumb_1_l": (
        (5.16822948431, 5.43732886333, 5.44157357104),
        (0.54119438212, 0.182029262149, 0.773035765762, 0.276386854494),
    ),
    "finger_thumb_2_l": (
        (4.31811664783, 7.46748480859, 7.83515030094),
        (0.541194380183, 0.182029207085, 0.773035766384, 0.276386892809),
    ),
    "finger_thumb_l_end": (
        (3.52167229259, 9.36947525525, 10.077618488),
        (0.541194380183, 0.182029207085, 0.773035766384, 0.276386892809),
    ),
}

REST_POS_CM: dict[str, np.ndarray] = {
    name: np.array(pos, dtype=np.float64) for name, (pos, _quat) in _TABLE.items()
}
"""node name -> (3,) float64 absolute translation in centimeters (verbatim)."""

REST_QUAT: dict[str, np.ndarray] = {
    name: np.array(quat, dtype=np.float64) for name, (_pos, quat) in _TABLE.items()
}
"""node name -> (4,) float64 absolute rotation quaternion, XYZW order (verbatim)."""

# --------------------------------------------------------------------------- #
# MediaPipe 21-landmark mapping. Metacarpal ("meta") nodes are model-only FK
# parents and are NOT MediaPipe landmarks; the thumb has no meta node.
#   0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
# --------------------------------------------------------------------------- #
MEDIAPIPE_NODES: list[str] = [
    "wrist_l",              # 0  wrist
    "finger_thumb_0_l",     # 1  thumb_cmc
    "finger_thumb_1_l",     # 2  thumb_mcp
    "finger_thumb_2_l",     # 3  thumb_ip
    "finger_thumb_l_end",   # 4  thumb_tip
    "finger_index_0_l",     # 5  index_mcp
    "finger_index_1_l",     # 6  index_pip
    "finger_index_2_l",     # 7  index_dip
    "finger_index_l_end",   # 8  index_tip
    "finger_middle_0_l",    # 9  middle_mcp
    "finger_middle_1_l",    # 10 middle_pip
    "finger_middle_2_l",    # 11 middle_dip
    "finger_middle_l_end",  # 12 middle_tip
    "finger_ring_0_l",      # 13 ring_mcp
    "finger_ring_1_l",      # 14 ring_pip
    "finger_ring_2_l",      # 15 ring_dip
    "finger_ring_l_end",    # 16 ring_tip
    "finger_pinky_0_l",     # 17 pinky_mcp
    "finger_pinky_1_l",     # 18 pinky_pip
    "finger_pinky_2_l",     # 19 pinky_dip
    "finger_pinky_l_end",   # 20 pinky_tip
]

# --------------------------------------------------------------------------- #
# Parent->child chains for FK. Each non-thumb finger is rooted at wrist_l via its
# fixed meta node; the thumb has no meta and hangs directly off the wrist.
# --------------------------------------------------------------------------- #
_CHAINS: list[list[str]] = [
    [
        "wrist_l",
        "finger_index_meta_l",
        "finger_index_0_l",
        "finger_index_1_l",
        "finger_index_2_l",
        "finger_index_l_end",
    ],
    [
        "wrist_l",
        "finger_middle_meta_l",
        "finger_middle_0_l",
        "finger_middle_1_l",
        "finger_middle_2_l",
        "finger_middle_l_end",
    ],
    [
        "wrist_l",
        "finger_ring_meta_l",
        "finger_ring_0_l",
        "finger_ring_1_l",
        "finger_ring_2_l",
        "finger_ring_l_end",
    ],
    [
        "wrist_l",
        "finger_pinky_meta_l",
        "finger_pinky_0_l",
        "finger_pinky_1_l",
        "finger_pinky_2_l",
        "finger_pinky_l_end",
    ],
    [
        "wrist_l",
        "finger_thumb_0_l",
        "finger_thumb_1_l",
        "finger_thumb_2_l",
        "finger_thumb_l_end",
    ],
]

SEGMENTS: list[tuple[str, str]] = [
    (chain[i], chain[i + 1]) for chain in _CHAINS for i in range(len(chain) - 1)
]
"""parent->child edges of the 5 finger chains (meta nodes included as parents)."""

# Left-handed Unity wire uses a negated x axis relative to the model frame.
_WIRE_AXIS = np.array([-1.0, 1.0, 1.0], dtype=np.float64)


def to_wire_meters(p_cm, wrist_cm) -> np.ndarray:
    """Model-frame position (cm) -> wrist-local, x-negated, meters (Unity wire).

    ``wire = ([-1, 1, 1] * (p_cm - wrist_cm)) / 100``. This is the exact
    model->wire transform the glove's protobuf/JSON stream uses; fk.py and the
    transcription cross-checks reuse it.
    """
    p = np.asarray(p_cm, dtype=np.float64)
    wrist = np.asarray(wrist_cm, dtype=np.float64)
    return (_WIRE_AXIS * (p - wrist)) / 100.0


# =========================================================================== #
# RIGHT-hand rest-pose skeleton.
#
# Transcribed verbatim from the vendor's "Hand Model and Data Specification"
# (S5), right-hand table. Same units/order as the LEFT ``_TABLE`` above
# (centimeters; XYZW quaternions). The LEFT names/tables above are UNCHANGED by
# this section -- existing importers of ``REST_POS_CM``/``REST_QUAT``/
# ``MEDIAPIPE_NODES``/``SEGMENTS`` keep getting the left skeleton exactly as
# before. Right-hand consumers (fk.py) use the ``_R`` names or the
# ``*_BY_SIDE`` dicts below.
# =========================================================================== #
_TABLE_R: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "Root": (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "RightHand": (
        (-8.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "wrist_r": (
        (-8.01597869955, -0.00319243990816, -0.062570899725),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "finger_index_meta_r": (
        (-7.86025399528, 2.10413001454, 1.41612997651),
        (-0.550753010824, 0.539577855624, 0.351434298812, 0.531056298718),
    ),
    "finger_index_0_r": (
        (-9.11996003172, 3.73569006174, 8.50216018147),
        (-0.464900669598, 0.513485228274, 0.443193991622, 0.569016145238),
    ),
    "finger_index_1_r": (
        (-9.46549001271, 3.61912006815, 12.8153999441),
        (-0.418738810367, 0.546730939299, 0.401463913545, 0.603796169934),
    ),
    "finger_index_2_r": (
        (-9.69119001003, 3.54297006905, 15.6328999107),
        (-0.401868146063, 0.53829823803, 0.40796017639, 0.618308575546),
    ),
    "finger_index_r_end": (
        (-9.87336001188, 3.48152006638, 17.9068999137),
        (-0.464900703591, 0.513485133683, 0.443194081541, 0.569016132789),
    ),
    "finger_middle_meta_r": (
        (-8.23370899819, 0.708761974936, 1.56930997968),
        (-0.533423185114, 0.472987949641, 0.419737279231, 0.561749696485),
    ),
    "finger_middle_0_r": (
        (-9.65287996638, 0.942902969689, 8.51039982412),
        (-0.493570447914, 0.518669359027, 0.463559282171, 0.52199913876),
    ),
    "finger_middle_1_r": (
        (-9.79164996487, 1.06379996837, 12.8172997772),
        (-0.487533304139, 0.487910867419, 0.492680618001, 0.530584650603),
    ),
    "finger_middle_2_r": (
        (-9.89872996178, 1.15708998353, 16.1408996965),
        (-0.493570505576, 0.518669433374, 0.463559195508, 0.521999087325),
    ),
    "finger_middle_r_end": (
        (-9.98206995874, 1.22970998089, 18.7277996023),
        (-0.493570505576, 0.518669433374, 0.463559195508, 0.521999087325),
    ),
    "finger_ring_meta_r": (
        (-8.06732100062, -0.657705024118, 1.57219997048),
        (-0.51669223092, 0.429887865514, 0.495547830323, 0.550143535319),
    ),
    "finger_ring_0_r": (
        (-8.98607300962, -1.32409003065, 8.07133003417),
        (-0.527700251867, 0.473873378439, 0.487278247149, 0.509447127033),
    ),
    "finger_ring_1_r": (
        (-9.29262001321, -1.30940003048, 12.0928000812),
        (-0.518613086083, 0.442107789446, 0.517618915142, 0.517350778623),
    ),
    "finger_ring_2_r": (
        (-9.50916001899, -1.29903003344, 14.9334001585),
        (-0.527700220878, 0.473873344733, 0.487278282022, 0.509447157128),
    ),
    "finger_ring_r_end": (
        (-9.67965002595, -1.29086003311, 17.1699002497),
        (-0.527700220878, 0.473873344733, 0.487278282022, 0.509447157128),
    ),
    "finger_pinky_meta_r": (
        (-7.76816300116, -1.90133000934, 1.45878997445),
        (-0.502233766644, 0.360574230945, 0.601636397582, 0.505748072389),
    ),
    "finger_pinky_0_r": (
        (-7.86898899845, -3.44991996759, 7.54976981024),
        (-0.565064692375, 0.413761381743, 0.543053388662, 0.463245539073),
    ),
    "finger_pinky_1_r": (
        (-8.07150439126, -3.55606996382, 10.5283997045),
        (-0.575649110471, 0.374918480934, 0.580690191527, 0.436878857044),
    ),
    "finger_pinky_2_r": (
        (-8.19337899197, -3.61994996156, 12.3209996989),
        (-0.668204946747, 0.32905173897, 0.587963212359, 0.315477991524),
    ),
    "finger_pinky_r_end": (
        (-8.31552099087, -3.68395996843, 14.1174996892),
        (-0.565064752026, 0.413761417283, 0.543053338237, 0.463245493681),
    ),
    "finger_thumb_0_r": (
        (-6.22459996305, 2.91460989392, 2.46726003289),
        (-0.276386849908, 0.773035739622, -0.182029268031, 0.541194419821),
    ),
    "finger_thumb_1_r": (
        (-5.16822992831, 5.4373299769, 5.44157013073),
        (-0.276386837157, 0.773035735333, -0.182029286244, 0.541194426333),
    ),
    "finger_thumb_2_r": (
        (-4.31811995431, 7.4674799148, 7.8351500575),
        (-0.276386777622, 0.773035715309, -0.182029371283, 0.541194456738),
    ),
    "finger_thumb_r_end": (
        (-3.52166995775, 9.36947990658, 10.0776000478),
        (-0.276386658894, 0.773035675375, -0.182029540873, 0.541194517372),
    ),
}

REST_POS_CM_R: dict[str, np.ndarray] = {
    name: np.array(pos, dtype=np.float64) for name, (pos, _quat) in _TABLE_R.items()
}
"""RIGHT-hand analogue of ``REST_POS_CM`` (node name -> (3,) cm position)."""

REST_QUAT_R: dict[str, np.ndarray] = {
    name: np.array(quat, dtype=np.float64) for name, (_pos, quat) in _TABLE_R.items()
}
"""RIGHT-hand analogue of ``REST_QUAT`` (node name -> (4,) XYZW quaternion)."""

MEDIAPIPE_NODES_R: list[str] = [
    "wrist_r",              # 0  wrist
    "finger_thumb_0_r",     # 1  thumb_cmc
    "finger_thumb_1_r",     # 2  thumb_mcp
    "finger_thumb_2_r",     # 3  thumb_ip
    "finger_thumb_r_end",   # 4  thumb_tip
    "finger_index_0_r",     # 5  index_mcp
    "finger_index_1_r",     # 6  index_pip
    "finger_index_2_r",     # 7  index_dip
    "finger_index_r_end",   # 8  index_tip
    "finger_middle_0_r",    # 9  middle_mcp
    "finger_middle_1_r",    # 10 middle_pip
    "finger_middle_2_r",    # 11 middle_dip
    "finger_middle_r_end",  # 12 middle_tip
    "finger_ring_0_r",      # 13 ring_mcp
    "finger_ring_1_r",      # 14 ring_pip
    "finger_ring_2_r",      # 15 ring_dip
    "finger_ring_r_end",    # 16 ring_tip
    "finger_pinky_0_r",     # 17 pinky_mcp
    "finger_pinky_1_r",     # 18 pinky_pip
    "finger_pinky_2_r",     # 19 pinky_dip
    "finger_pinky_r_end",   # 20 pinky_tip
]
"""RIGHT-hand analogue of ``MEDIAPIPE_NODES``."""

_CHAINS_R: list[list[str]] = [
    [
        "wrist_r",
        "finger_index_meta_r",
        "finger_index_0_r",
        "finger_index_1_r",
        "finger_index_2_r",
        "finger_index_r_end",
    ],
    [
        "wrist_r",
        "finger_middle_meta_r",
        "finger_middle_0_r",
        "finger_middle_1_r",
        "finger_middle_2_r",
        "finger_middle_r_end",
    ],
    [
        "wrist_r",
        "finger_ring_meta_r",
        "finger_ring_0_r",
        "finger_ring_1_r",
        "finger_ring_2_r",
        "finger_ring_r_end",
    ],
    [
        "wrist_r",
        "finger_pinky_meta_r",
        "finger_pinky_0_r",
        "finger_pinky_1_r",
        "finger_pinky_2_r",
        "finger_pinky_r_end",
    ],
    [
        "wrist_r",
        "finger_thumb_0_r",
        "finger_thumb_1_r",
        "finger_thumb_2_r",
        "finger_thumb_r_end",
    ],
]

SEGMENTS_R: list[tuple[str, str]] = [
    (chain[i], chain[i + 1]) for chain in _CHAINS_R for i in range(len(chain) - 1)
]
"""RIGHT-hand analogue of ``SEGMENTS``."""

# --------------------------------------------------------------------------- #
# Side-keyed accessors. LEFT entries are exactly the pre-existing left-only
# names above (unchanged); RIGHT entries are the ``_R`` tables just above.
# fk.py (side-aware FK) indexes these by ``side`` instead of duplicating the
# left/right selection logic.
# --------------------------------------------------------------------------- #
REST_POS_CM_BY_SIDE: dict[str, dict[str, np.ndarray]] = {
    "left": REST_POS_CM,
    "right": REST_POS_CM_R,
}
REST_QUAT_BY_SIDE: dict[str, dict[str, np.ndarray]] = {
    "left": REST_QUAT,
    "right": REST_QUAT_R,
}
MEDIAPIPE_NODES_BY_SIDE: dict[str, list[str]] = {
    "left": MEDIAPIPE_NODES,
    "right": MEDIAPIPE_NODES_R,
}
SEGMENTS_BY_SIDE: dict[str, list[tuple[str, str]]] = {
    "left": SEGMENTS,
    "right": SEGMENTS_R,
}
