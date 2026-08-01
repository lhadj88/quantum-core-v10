#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent
PARTS = sorted((ROOT / "payload_parts").glob("part_*.b64"))
if len(PARTS) != 2:
    raise RuntimeError(f"Expected 2 R19 payload parts, found {len(PARTS)}")
encoded = "".join(p.read_text(encoding="utf-8").strip() for p in PARTS)
blob = base64.b64decode(encoded, validate=True)
archive = ROOT / "r19_payload.tar.gz"
archive.write_bytes(blob)
with tarfile.open(archive, "r:gz") as tf:
    names = {pathlib.PurePosixPath(m.name).name for m in tf.getmembers() if m.isfile()}
    if names != {"run_r19.py", "requirements.txt"}:
        raise RuntimeError(f"Unexpected R19 payload members: {sorted(names)}")
    for member in tf.getmembers():
        target = (ROOT / member.name).resolve()
        if ROOT.resolve() not in target.parents and target != ROOT.resolve():
            raise RuntimeError(f"Unsafe payload path: {member.name}")
    tf.extractall(ROOT)
print(f"R19 payload reconstructed: {len(blob)} bytes")
