#!/usr/bin/env python3
from __future__ import annotations

import base64
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent
PARTS = sorted((ROOT / "payload_parts").glob("part_*.b64"))
if len(PARTS) != 7:
    raise RuntimeError(f"Expected 7 payload parts, found {len(PARTS)}")

encoded = "".join(part.read_text(encoding="utf-8").strip() for part in PARTS)
blob = base64.b64decode(encoded, validate=True)
archive = ROOT / "r18_payload.tar.gz"
archive.write_bytes(blob)

with tarfile.open(archive, "r:gz") as tf:
    members = tf.getmembers()
    allowed = {"run_r18.py", "events_2021_2026.csv"}
    names = {pathlib.PurePosixPath(member.name).name for member in members if member.isfile()}
    if names != allowed:
        raise RuntimeError(f"Unexpected R18 payload members: {sorted(names)}")
    for member in members:
        target = (ROOT / member.name).resolve()
        if ROOT.resolve() not in target.parents and target != ROOT.resolve():
            raise RuntimeError(f"Unsafe payload path: {member.name}")
    tf.extractall(ROOT)

print(f"R18 payload reconstructed: {archive.stat().st_size} bytes")
