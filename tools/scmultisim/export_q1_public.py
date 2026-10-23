"""Export validated public RNA only; the unchanged manifest may reference private hashes.

The 289 observation files and manifest are verified by the learner's loader.
Private files, simulator states and true networks are never included or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from grn_experiment.q1_pilot_data import (
    _directory_payload, _MAX_MANIFEST_BYTES, _MAX_OBSERVATION_BYTES, load_modular_q1,
)


def export(dataset: Path, output: Path) -> dict:
    """Validate the complete public dataset and exclusively publish a new archive."""
    dataset, output = Path(dataset).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; existing archives are never overwritten")
    if output.resolve().is_relative_to(dataset):
        raise ValueError("Output must be outside the input dataset directory")
    data = load_modular_q1(dataset)
    whitelist = data.provenance["verified_observation_sha256"]
    manifest_hash = data.provenance["manifest_sha256"]
    del data
    entries = [("manifest.json", manifest_hash, _MAX_MANIFEST_BYTES)]
    for relative, checksum in sorted(whitelist.items()):
        path = PurePosixPath(relative)
        if path.parent != PurePosixPath("observations/q1") or path.suffix != ".npz":
            raise ValueError("Only allowlisted observations/q1 NPZ files may be exported")
        entries.append((relative, checksum, _MAX_OBSERVATION_BYTES))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".q1-public-", delete=False) as stream:
            temporary = Path(stream.name)
            with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
                for relative, checksum, limit in entries:
                    payload = _directory_payload(dataset, relative, limit)
                    if hashlib.sha256(payload).hexdigest() != checksum:
                        raise ValueError(f"Payload changed after validation: {relative}")
                    member = tarfile.TarInfo(f"q1-modular-64/{relative}")
                    member.size, member.mode = len(payload), 0o644
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ""
                    member.type = tarfile.REGTYPE
                    archive.addfile(member, io.BytesIO(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic publication without replacing an existing path.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with output.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": checksum,
            "members": len(entries), "private_files_included": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.dataset, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
