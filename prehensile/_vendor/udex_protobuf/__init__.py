"""Vendored, protoc-generated HandDriver protobuf modules (third-party; DO NOT EDIT).

``handdriver_teleop_pb2.py`` was generated with a FLAT cross-import
(``import handdriver_algebra_pb2 as handdriver__algebra__pb2``) rather than a
package-relative one, so its own directory must be on ``sys.path`` under that
bare name for the generated code to import. Now that these modules live inside
the ``prehensile`` package instead of a top-level ``vendor/`` tree, we put
THIS directory on ``sys.path`` (once) here, so the flat cross-import resolves
no matter who imports ``handdriver_teleop_pb2`` -- callers no longer need
their own path shim.

The proper long-term fix is to regenerate both modules from the in-tree
``.proto`` sources with ``grpcio-tools`` (modern protoc/grpcio-tools emit
package-relative imports), which would let this shim go away entirely.
Deliberately not done as part of this change.
"""

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
