"""Glove -> REALHAND L6 teleoperation pipeline, end to end.

Two glove sources are wired behind the same ``poll() -> (21,3)`` MediaPipe
keypoint seam: the UDCap glove over UDP (``udcap.py``, decoded via ``fk.py``'s
per-finger quaternion forward kinematics) and the Wuji SDK glove (``wuji.py``,
whose SDK import stays lazy so the module loads fine without it installed).
Two angle mappings turn those keypoints into the L6's 6 channels: the
``dex_retargeting``-based vector optimizer (``retarget.py``) and a direct
geometric curl map (``curlmap.py``) that needs no optimizer at all. A MuJoCo
viewer (``viz.py``) previews either mapping in sim, and the live CLI
(``teleop.py``) drives the real hand over CAN via the L6 command path
(``command.py``, ``l6_discovery.py``).

Also home to the pipeline's supporting pieces: per-glove/per-hand profiles
(``profiles.py``), curl-map tuning (``tuning.py``), the rest-pose skeleton
(``rest_pose.py``), and an offline plumbing check (``offline_check.py``).

The repo's ops tools sit alongside it: ``tools/bring_up_hand.py`` names the
CAN adapters and brings the link up, ``tools/probe_udp.py`` checks the glove
wire, and ``tools/record_thumb_sweep.py`` captures the sweeps behind the
curl-map thumb tuning in ``configs/curl_tuning.yml``.

Module-level imports here stay absent on purpose: ``profiles`` imports
``wuji`` lazily from inside a function, so a deployment that only needs the
UDCap path never has to install a glove SDK. Importing anything here would
defeat that."""
