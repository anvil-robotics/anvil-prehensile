import socket
import argparse
import sys
from pathlib import Path

# Ensure we can import the generated proto files
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import handdriver_teleop_pb2

from coordinate_system import (
    convert_teleop_data_pose_in_place,
    convert_teleop_data_quat_in_place,
    coordinate_system_spec_help,
    coordinate_system_to_string,
    handdriver_default_coordinate_system,
    parse_coordinate_system_vuh,
)


OUTPUT_DIR = PROJECT_DIR / "Tool"
SOURCE_OUTPUT_FILE = "Source.txt"
CONVERTED_OUTPUT_FILE = "Coordinate conversion.txt"


def prompt_text(label, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    return default


def prompt_int(label, default):
    while True:
        value = prompt_text(label, str(default))
        try:
            return int(value)
        except (TypeError, ValueError):
            print("Please enter a valid integer.")


def prompt_choice(label, choices, default):
    choices_text = "/".join(choices)
    while True:
        value = prompt_text(f"{label} ({choices_text})", default)
        if value in choices:
            return value
        print(f"Please enter one of: {choices_text}")


def prompt_bool(label, default=False):
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} ({suffix}): ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "true", "1"):
            return True
        if value in ("n", "no", "false", "0"):
            return False
        print("Please enter y or n.")


def collect_manual_args(args):
    print("Manual input mode")
    args.port = prompt_int("UDP port", args.port)
    args.type = prompt_choice("Data type", ["angle", "quat", "pose"], args.type)
    args.source_coord = (
        prompt_text("Source coordinate system spec (empty = HandDriver default)", args.source_coord or "")
        or None
    )

    if args.type == "angle":
        args.target_coord = None
        print("Coordinate conversion is skipped for angle data.")
    else:
        args.target_coord = (
            prompt_text("Target coordinate system spec (empty = no conversion)", args.target_coord or "")
            or None
        )

    args.write_output = prompt_bool("Append printed data to Tool files", args.write_output)
    return args


def format_float_list(values):
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def build_single_hand_text(name, hand):
    hand_text = str(hand).rstrip()
    if hand_text:
        return f"{name}:\n{hand_text}"
    return f"{name}:"


def build_hand_data_text(msg):
    return "\n".join(
        [
            f"Teleop Header: TimeStamp={msg.TimeStamp}, "
            f"FrameIndex={msg.FrameIndex}, RoleName={msg.RoleName}",
            build_single_hand_text("LeftHand", msg.LeftHand),
            build_single_hand_text("RightHand", msg.RightHand),
        ]
    )


def get_output_path(converted, output_dir=OUTPUT_DIR):
    output_name = CONVERTED_OUTPUT_FILE if converted else SOURCE_OUTPUT_FILE
    return Path(output_dir) / output_name


def append_output(output_path, text):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(text)
        output_file.write("\n")


def apply_coordinate_conversion(msg, data_type, source_coord, target_coord):
    if target_coord is None:
        return msg

    if data_type == "pose":
        return convert_teleop_data_pose_in_place(msg, source_coord, target_coord)
    if data_type == "quat":
        return convert_teleop_data_quat_in_place(msg, source_coord, target_coord)

    raise ValueError("coordinate conversion is only supported for quat and pose data")


def main():
    parser = argparse.ArgumentParser(description='Receive Proto UDP data')
    parser.add_argument('--port', type=int, default=5555, help='UDP port to listen on')
    parser.add_argument('--type', type=str, choices=['angle', 'quat', 'pose'], default='angle', help='Expected data type')
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Prompt for receiver settings before listening.',
    )
    parser.add_argument(
        '--source-coord',
        type=str,
        default=None,
        help='Source coordinate system spec. Defaults to HandDriver left,+y,zfrom,1.',
    )
    parser.add_argument(
        '--target-coord',
        type=str,
        default=None,
        help='Target coordinate system spec. When set, quat/pose data is converted before printing.',
    )
    parser.add_argument(
        '--write-output',
        action='store_true',
        help='Append printed data to Tool/Source.txt or Tool/Coordinate conversion.txt.',
    )
    parser.add_argument(
        '--coord-help',
        action='store_true',
        help='Show coordinate system spec help and exit.',
    )
    args = parser.parse_args()

    if args.coord_help:
        print(coordinate_system_spec_help())
        return

    if args.manual or len(sys.argv) == 1:
        args = collect_manual_args(args)

    try:
        source_coord = (
            parse_coordinate_system_vuh(args.source_coord)
            if args.source_coord
            else handdriver_default_coordinate_system()
        )
        target_coord = parse_coordinate_system_vuh(args.target_coord) if args.target_coord else None
    except ValueError as e:
        print(f"Invalid coordinate system spec: {e}")
        print(coordinate_system_spec_help())
        return

    if target_coord is not None and args.type == "angle":
        print("Coordinate conversion is only supported for --type quat or --type pose.")
        return

    output_path = get_output_path(converted=target_coord is not None) if args.write_output else None

    message_type_map = {
        'angle': handdriver_teleop_pb2.TeleopDataAngle,
        'quat': handdriver_teleop_pb2.TeleopDataQuat,
        'pose': handdriver_teleop_pb2.TeleopDataPose,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('0.0.0.0', args.port))
        print(f"Listening on UDP port {args.port} for {args.type} data...")
        if output_path is not None:
            print(f"Writing received data to {output_path}")
        else:
            print("File writing is disabled. Use --write-output to append printed data to Tool files.")
        if target_coord is not None:
            print(f"Source coordinate system: {coordinate_system_to_string(source_coord)}")
            print(f"Target coordinate system: {coordinate_system_to_string(target_coord)}")
    except Exception as e:
        print(f"Error binding to port {args.port}: {e}")
        return

    while True:
        try:
            data, addr = sock.recvfrom(65535)

            try:
                msg = message_type_map[args.type]()
                msg.ParseFromString(data)
                apply_coordinate_conversion(msg, args.type, source_coord, target_coord)
                output_text = (
                    f"\nReceived {len(data)} bytes from {addr}\n"
                    f"{build_hand_data_text(msg)}"
                )
                print(output_text)
                if output_path is not None:
                    append_output(output_path, output_text)
            except Exception as e:
                print(f"Failed to parse protobuf message: {e}")
                
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error receiving data: {e}")

if __name__ == "__main__":
    main()
