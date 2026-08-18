from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import handdriver_algebra_pb2
import handdriver_teleop_pb2


_EPSILON = 1e-8

_Vec3d = Tuple[float, float, float]
_Mat3 = List[List[float]]


class Side(Enum):
    Left = "left"
    Right = "right"


class AxisPolarity(Enum):
    NegativeZ = "-Z"
    NegativeY = "-Y"
    NegativeX = "-X"
    PositiveX = "+X"
    PositiveY = "+Y"
    PositiveZ = "+Z"


class AxisView(Enum):
    ZFromViewer = "ZFromViewer"
    YFromViewer = "YFromViewer"
    XFromViewer = "XFromViewer"
    XToViewer = "XToViewer"
    YToViewer = "YToViewer"
    ZToViewer = "ZToViewer"


@dataclass
class CoordinateSystemVUH:
    view: AxisView = AxisView.ZFromViewer
    up: AxisPolarity = AxisPolarity.PositiveY
    handedness: Side = Side.Left
    unitScale: float = 1.0


def handdriver_default_coordinate_system() -> CoordinateSystemVUH:
    return CoordinateSystemVUH(
        view=AxisView.ZFromViewer,
        up=AxisPolarity.PositiveY,
        handedness=Side.Left,
        unitScale=1.0,
    )


def validate_coordinate_system(coordinate_system: CoordinateSystemVUH) -> None:
    if coordinate_system.unitScale <= 0.0 or not math.isfinite(coordinate_system.unitScale):
        raise ValueError("unitScale must be a finite positive number")

    view_axis = _axis_from_view(coordinate_system.view)
    up_axis = _axis_from_polarity(coordinate_system.up)
    if view_axis < 0 or up_axis < 0:
        raise ValueError("view and up must be valid axes")

    if view_axis == up_axis:
        raise ValueError("view and up cannot use the same axis")


def is_coordinate_system_valid(coordinate_system: CoordinateSystemVUH) -> bool:
    try:
        validate_coordinate_system(coordinate_system)
        return True
    except ValueError:
        return False


def parse_coordinate_system_vuh(spec: str) -> CoordinateSystemVUH:
    parsed = handdriver_default_coordinate_system()
    seen_side = False
    seen_up = False
    seen_view = False
    seen_scale = False

    tokens = _split_spec(spec)
    if not tokens:
        raise ValueError("coordinate system spec is empty")

    for raw_token in tokens:
        key = ""
        value = raw_token
        if "=" in raw_token:
            key, value = raw_token.split("=", 1)
            key = _compact_token(key)

        if key in ("handedness", "hand", "side", "h"):
            parsed.handedness = _parse_side_value(value)
            seen_side = True
            continue

        if key in ("up", "u"):
            parsed.up = _parse_axis_polarity_value(value)
            seen_up = True
            continue

        if key in ("view", "v", "forward"):
            parsed.view = _parse_axis_view_value(value)
            seen_view = True
            continue

        if key in ("scale", "unitscale", "unit"):
            parsed.unitScale = _parse_scale_value(value)
            seen_scale = True
            continue

        if not seen_side:
            side = _try_parse_side_value(value)
            if side is not None:
                parsed.handedness = side
                seen_side = True
                continue

        if not seen_up:
            up = _try_parse_axis_polarity_value(value)
            if up is not None:
                parsed.up = up
                seen_up = True
                continue

        if not seen_view:
            view = _try_parse_axis_view_value(value)
            if view is not None:
                parsed.view = view
                seen_view = True
                continue

        if not seen_scale:
            scale = _try_parse_scale_value(value)
            if scale is not None:
                parsed.unitScale = scale
                seen_scale = True
                continue

        raise ValueError(f"unrecognized coordinate system token: {raw_token}")

    validate_coordinate_system(parsed)
    return parsed


def coordinate_system_to_string(coordinate_system: CoordinateSystemVUH) -> str:
    return (
        f"{coordinate_system.handedness.value}"
        f",up={coordinate_system.up.value}"
        f",view={coordinate_system.view.value}"
        f",unitScale={coordinate_system.unitScale:g}"
    )


def coordinate_system_spec_help() -> str:
    return (
        "Coordinate spec format:\n"
        '  right,+z,xfrom,1\n'
        '  handedness=left,up=+y,view=zfrom,unit=0.01\n'
        "\n"
        "Fields match MANUS CoordinateSystemVUH:\n"
        "  handedness: right | left\n"
        "  up: +x | -x | +y | -y | +z | -z\n"
        "  view: xfrom | yfrom | zfrom | xto | yto | zto\n"
        "  unitScale: 1.0 meters, 0.01 centimeters, 0.001 millimeters\n"
        "\n"
        "Default source is HandDriver: left,+y,zfrom,1."
    )


def convert_position(
    position: handdriver_algebra_pb2.Vec3,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_algebra_pb2.Vec3:
    source_basis = _build_basis(source)
    target_basis = _build_basis(target)
    target_inverse = _transpose(target_basis)

    source_vector = (
        float(position.x) * source.unitScale,
        float(position.y) * source.unitScale,
        float(position.z) * source.unitScale,
    )
    canonical_vector = _mat_vec_multiply(source_basis, source_vector)
    target_vector = _mat_vec_multiply(target_inverse, canonical_vector)

    return handdriver_algebra_pb2.Vec3(
        x=target_vector[0] / target.unitScale,
        y=target_vector[1] / target.unitScale,
        z=target_vector[2] / target.unitScale,
    )


def convert_rotation(
    rotation: handdriver_algebra_pb2.Quat,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_algebra_pb2.Quat:
    length_squared = (
        rotation.x * rotation.x
        + rotation.y * rotation.y
        + rotation.z * rotation.z
        + rotation.w * rotation.w
    )
    if length_squared < _EPSILON:
        result = handdriver_algebra_pb2.Quat()
        result.CopyFrom(rotation)
        return result

    source_basis = _build_basis(source)
    target_basis = _build_basis(target)
    source_inverse = _transpose(source_basis)
    target_inverse = _transpose(target_basis)

    source_rotation = _quaternion_to_matrix(rotation)
    canonical_rotation = _mat_multiply(
        _mat_multiply(source_basis, source_rotation),
        source_inverse,
    )
    target_rotation = _mat_multiply(
        _mat_multiply(target_inverse, canonical_rotation),
        target_basis,
    )

    return _matrix_to_quaternion(target_rotation)


def convert_joint_pose(
    pose: handdriver_teleop_pb2.JointPose,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_teleop_pb2.JointPose:
    result = handdriver_teleop_pb2.JointPose()
    result.CopyFrom(pose)
    result.globalPosition.CopyFrom(convert_position(pose.globalPosition, source, target))
    result.globalRotation.CopyFrom(convert_rotation(pose.globalRotation, source, target))
    return result


def convert_glove_hand_data_pose_in_place(
    hand: handdriver_teleop_pb2.GloveHandDataPose,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_teleop_pb2.GloveHandDataPose:
    if hand is None:
        return hand

    hand.imu.CopyFrom(convert_rotation(hand.imu, source, target))
    for pose in hand.pose:
        pose.globalPosition.CopyFrom(convert_position(pose.globalPosition, source, target))
        pose.globalRotation.CopyFrom(convert_rotation(pose.globalRotation, source, target))
    return hand


def convert_teleop_data_pose_in_place(
    data: handdriver_teleop_pb2.TeleopDataPose,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_teleop_pb2.TeleopDataPose:
    if data is None:
        return data

    convert_glove_hand_data_pose_in_place(data.LeftHand, source, target)
    convert_glove_hand_data_pose_in_place(data.RightHand, source, target)
    return data


def convert_glove_hand_data_quat_in_place(
    hand: handdriver_teleop_pb2.GloveHandDataQuat,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_teleop_pb2.GloveHandDataQuat:
    if hand is None:
        return hand

    hand.imu.CopyFrom(convert_rotation(hand.imu, source, target))
    for joint in hand.joints:
        joint.CopyFrom(convert_rotation(joint, source, target))
    return hand


def convert_teleop_data_quat_in_place(
    data: handdriver_teleop_pb2.TeleopDataQuat,
    source: CoordinateSystemVUH,
    target: CoordinateSystemVUH,
) -> handdriver_teleop_pb2.TeleopDataQuat:
    if data is None:
        return data

    convert_glove_hand_data_quat_in_place(data.LeftHand, source, target)
    convert_glove_hand_data_quat_in_place(data.RightHand, source, target)
    return data


def _trim(value: str) -> str:
    return value.strip(" \t\r\n")


def _compact_token(value: str) -> str:
    trimmed_value = _trim(value)
    result = []
    for index, ch in enumerate(trimmed_value):
        if ch in ("_", "-", " ", "\t"):
            if ch == "-" and (index == 0 or index + 1 == len(trimmed_value)):
                result.append(ch)
            continue
        result.append(ch.lower())

    token = "".join(result)
    for prefix in ("axispolarity", "axisview", "side"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def _split_spec(spec: str) -> List[str]:
    tokens = []
    current = []
    for ch in spec:
        if ch in (",", ";"):
            token = _trim("".join(current))
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(ch)

    token = _trim("".join(current))
    if token:
        tokens.append(token)
    return tokens


def _try_parse_side_value(value: str) -> Optional[Side]:
    token = _compact_token(value)
    if token in ("left", "l", "lefthanded"):
        return Side.Left
    if token in ("right", "r", "righthanded"):
        return Side.Right
    return None


def _parse_side_value(value: str) -> Side:
    side = _try_parse_side_value(value)
    if side is None:
        raise ValueError(f"invalid handedness value: {value}")
    return side


def _try_parse_axis_polarity_value(value: str) -> Optional[AxisPolarity]:
    token = _compact_token(value)
    if token in ("+x", "x+", "positivex", "xpositive", "xup"):
        return AxisPolarity.PositiveX
    if token in ("+y", "y+", "positivey", "ypositive", "yup"):
        return AxisPolarity.PositiveY
    if token in ("+z", "z+", "positivez", "zpositive", "zup"):
        return AxisPolarity.PositiveZ
    if token in ("-x", "x-", "negativex", "xnegative", "xdown"):
        return AxisPolarity.NegativeX
    if token in ("-y", "y-", "negativey", "ynegative", "ydown"):
        return AxisPolarity.NegativeY
    if token in ("-z", "z-", "negativez", "znegative", "zdown"):
        return AxisPolarity.NegativeZ
    return None


def _parse_axis_polarity_value(value: str) -> AxisPolarity:
    polarity = _try_parse_axis_polarity_value(value)
    if polarity is None:
        raise ValueError(f"invalid up value: {value}")
    return polarity


def _try_parse_axis_view_value(value: str) -> Optional[AxisView]:
    token = _compact_token(value)
    if token in ("xfromviewer", "xfrom", "xaway", "xforward", "+x", "x+"):
        return AxisView.XFromViewer
    if token in ("yfromviewer", "yfrom", "yaway", "yforward", "+y", "y+"):
        return AxisView.YFromViewer
    if token in ("zfromviewer", "zfrom", "zaway", "zforward", "+z", "z+"):
        return AxisView.ZFromViewer
    if token in (
        "xtoviewer",
        "xto",
        "xtowardviewer",
        "xtowardsviewer",
        "xback",
        "-x",
        "x-",
    ):
        return AxisView.XToViewer
    if token in (
        "ytoviewer",
        "yto",
        "ytowardviewer",
        "ytowardsviewer",
        "yback",
        "-y",
        "y-",
    ):
        return AxisView.YToViewer
    if token in (
        "ztoviewer",
        "zto",
        "ztowardviewer",
        "ztowardsviewer",
        "zback",
        "-z",
        "z-",
    ):
        return AxisView.ZToViewer
    return None


def _parse_axis_view_value(value: str) -> AxisView:
    view = _try_parse_axis_view_value(value)
    if view is None:
        raise ValueError(f"invalid view value: {value}")
    return view


def _try_parse_scale_value(value: str) -> Optional[float]:
    try:
        result = float(_trim(value))
    except ValueError:
        return None

    if result <= 0.0 or not math.isfinite(result):
        return None
    return result


def _parse_scale_value(value: str) -> float:
    scale = _try_parse_scale_value(value)
    if scale is None:
        raise ValueError(f"invalid unitScale value: {value}")
    return scale


def _axis_from_polarity(polarity: AxisPolarity) -> int:
    if polarity in (AxisPolarity.NegativeX, AxisPolarity.PositiveX):
        return 0
    if polarity in (AxisPolarity.NegativeY, AxisPolarity.PositiveY):
        return 1
    if polarity in (AxisPolarity.NegativeZ, AxisPolarity.PositiveZ):
        return 2
    return -1


def _axis_from_view(view: AxisView) -> int:
    if view in (AxisView.XFromViewer, AxisView.XToViewer):
        return 0
    if view in (AxisView.YFromViewer, AxisView.YToViewer):
        return 1
    if view in (AxisView.ZFromViewer, AxisView.ZToViewer):
        return 2
    return -1


def _canonical_up_from_polarity(polarity: AxisPolarity) -> _Vec3d:
    if polarity in (
        AxisPolarity.PositiveX,
        AxisPolarity.PositiveY,
        AxisPolarity.PositiveZ,
    ):
        return (0.0, 1.0, 0.0)
    if polarity in (
        AxisPolarity.NegativeX,
        AxisPolarity.NegativeY,
        AxisPolarity.NegativeZ,
    ):
        return (0.0, -1.0, 0.0)
    return (0.0, 0.0, 0.0)


def _canonical_forward_from_view(view: AxisView) -> _Vec3d:
    if view in (AxisView.XFromViewer, AxisView.YFromViewer, AxisView.ZFromViewer):
        return (0.0, 0.0, 1.0)
    if view in (AxisView.XToViewer, AxisView.YToViewer, AxisView.ZToViewer):
        return (0.0, 0.0, -1.0)
    return (0.0, 0.0, 0.0)


def _cross(a: _Vec3d, b: _Vec3d) -> _Vec3d:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _negate(value: _Vec3d) -> _Vec3d:
    return (-value[0], -value[1], -value[2])


def _set_column(matrix: _Mat3, column: int, value: _Vec3d) -> None:
    matrix[0][column] = value[0]
    matrix[1][column] = value[1]
    matrix[2][column] = value[2]


def _transpose(matrix: _Mat3) -> _Mat3:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _mat_multiply(lhs: _Mat3, rhs: _Mat3) -> _Mat3:
    result = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for column in range(3):
            for k in range(3):
                result[row][column] += lhs[row][k] * rhs[k][column]
    return result


def _mat_vec_multiply(matrix: _Mat3, vector: _Vec3d) -> _Vec3d:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _build_basis(coordinate_system: CoordinateSystemVUH) -> _Mat3:
    validate_coordinate_system(coordinate_system)

    basis = [[0.0, 0.0, 0.0] for _ in range(3)]
    view_axis = _axis_from_view(coordinate_system.view)
    up_axis = _axis_from_polarity(coordinate_system.up)
    third_axis = 3 - view_axis - up_axis

    forward = _canonical_forward_from_view(coordinate_system.view)
    up = _canonical_up_from_polarity(coordinate_system.up)
    third = _cross(forward, up)
    if coordinate_system.handedness == Side.Left:
        third = _negate(third)

    _set_column(basis, view_axis, forward)
    _set_column(basis, up_axis, up)
    _set_column(basis, third_axis, third)
    return basis


def _quaternion_to_matrix(rotation: handdriver_algebra_pb2.Quat) -> _Mat3:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)
    length_squared = x * x + y * y + z * z + w * w

    result = [[0.0, 0.0, 0.0] for _ in range(3)]
    if length_squared < _EPSILON:
        return result

    inv_length = 1.0 / math.sqrt(length_squared)
    nx = x * inv_length
    ny = y * inv_length
    nz = z * inv_length
    nw = w * inv_length

    xx = nx * nx
    yy = ny * ny
    zz = nz * nz
    xy = nx * ny
    xz = nx * nz
    yz = ny * nz
    wx = nw * nx
    wy = nw * ny
    wz = nw * nz

    result[0][0] = 1.0 - 2.0 * (yy + zz)
    result[0][1] = 2.0 * (xy - wz)
    result[0][2] = 2.0 * (xz + wy)
    result[1][0] = 2.0 * (xy + wz)
    result[1][1] = 1.0 - 2.0 * (xx + zz)
    result[1][2] = 2.0 * (yz - wx)
    result[2][0] = 2.0 * (xz - wy)
    result[2][1] = 2.0 * (yz + wx)
    result[2][2] = 1.0 - 2.0 * (xx + yy)
    return result


def _matrix_to_quaternion(matrix: _Mat3) -> handdriver_algebra_pb2.Quat:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2][1] - matrix[1][2]) / s
        y = (matrix[0][2] - matrix[2][0]) / s
        z = (matrix[1][0] - matrix[0][1]) / s
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / s
        x = 0.25 * s
        y = (matrix[0][1] + matrix[1][0]) / s
        z = (matrix[0][2] + matrix[2][0]) / s
    elif matrix[1][1] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / s
        x = (matrix[0][1] + matrix[1][0]) / s
        y = 0.25 * s
        z = (matrix[1][2] + matrix[2][1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / s
        x = (matrix[0][2] + matrix[2][0]) / s
        y = (matrix[1][2] + matrix[2][1]) / s
        z = 0.25 * s

    length_squared = x * x + y * y + z * z + w * w
    if length_squared >= _EPSILON:
        inv_length = 1.0 / math.sqrt(length_squared)
        x *= inv_length
        y *= inv_length
        z *= inv_length
        w *= inv_length

    return handdriver_algebra_pb2.Quat(x=x, y=y, z=z, w=w)
