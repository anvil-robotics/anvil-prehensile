"""Shared UDP helpers for the discovery tool (probe_udp.py).

Small, stdlib-only-at-import module. Two jobs:

  * ``bind_udp(port)`` - the common receiver socket + EADDRINUSE handling the
    tool needs.
  * ``load_pb2()`` - imports the vendored, regenerated ``handdriver_teleop_pb2``
    from ``prehensile._vendor.udex_protobuf``. Imported *lazily* here (inside
    the function) so that merely importing this module stays stdlib-only (a
    caller that only records raw datagrams, without decoding them, never
    needs to parse protobuf at all).

A package module (``prehensile.udexio``), so both core code and the research
``tools/`` scripts import it normally -- ``from prehensile import udexio`` --
instead of via a sibling-file sys.path hack.
"""

import errno
import socket

RECV_BUFSIZE = 65535


def load_pb2():
    """Import and return the regenerated ``handdriver_teleop_pb2`` module.

    Imports from ``prehensile._vendor.udex_protobuf``, whose ``__init__.py``
    puts its own directory on ``sys.path`` so the generated module's flat
    cross-import (``import handdriver_algebra_pb2``) resolves. Raises
    ImportError if the module (or its protobuf runtime) cannot be loaded;
    callers that only need JSON analysis may catch that and carry on.
    """
    from prehensile._vendor.udex_protobuf import handdriver_teleop_pb2

    return handdriver_teleop_pb2


def bind_udp(port):
    """Return a UDP socket bound to 0.0.0.0:port, or exit 1 with guidance.

    We deliberately do NOT set SO_REUSEADDR/SO_REUSEPORT: a second binder
    *should* fail loudly so the operator notices a stale receiver rather than
    silently splitting the datagram stream.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        sock.close()
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"error: UDP port {port} is already in use (EADDRINUSE).\n"
                f"Something else is already bound to it (a stale receiver, or "
                f"another copy of this tool).\nFind the owner with:\n"
                f"    ss -ulpn 'sport = :{port}'\n"
                f"then stop it, or run with a different --port."
            )
        raise
    return sock
