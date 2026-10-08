"""Compile and run the pinned, unmodified GNW JAR; never install dependencies.

Runtime artifacts and generated acceptance data stay out of source control.
This deliberately exercises only the fixed four-gene acceptance network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


SOURCE_COMMIT = "c5310349f5d5723306585c2bb62aedbdeb70db46"
JAR_SHA256 = "b44ce5fcc95d1efc882d0da8a2b0428df4b175fbe4f0a836b4e66e120c08be5b"
JAR_URL = (
    "https://raw.githubusercontent.com/tschaffter/genenetweaver/"
    f"{SOURCE_COMMIT}/gnw-3.1.2b.jar"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(command: list[str], log: Path, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--gnw-jar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-check", action="store_true", help="Generate only; not acceptance success")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    jar = args.gnw_jar.resolve()
    java_home = args.java_home.resolve()
    if output == root or root in output.parents and output.parts[len(root.parts)] != "results":
        parser.error("Within this repository, output must be below the ignored results/ directory")
    if sha256(jar) != JAR_SHA256:
        parser.error("GNW JAR checksum differs from the pinned official artifact")
    # A new output directory is preferred; avoid silently overwriting user results.
    generated = ("trajectories.csv", "refined_trajectories.csv", "report.json")
    if any((output / name).exists() for name in generated):
        parser.error("Output already contains results; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "truth").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    classes = output / "classes"
    classes.mkdir(exist_ok=True)
    source = Path(__file__).with_name("GnwAcceptance.java")
    java = str(java_home / "bin" / "java")
    javac = str(java_home / "bin" / "javac")
    version = execute([java, "-version"], output / "logs" / "java_version.log", root)
    compile_command = [javac, "-cp", str(jar), "-d", str(classes), str(source)]
    execute(compile_command, output / "logs" / "compile.log", root)
    base_command = [java, "-ea", "-Djava.awt.headless=true", "-cp",
                    os.pathsep.join((str(classes), str(jar))), "GnwAcceptance"]
    execute(base_command + [str(output)], output / "logs" / "native_run.log", root)
    # A second native run with exactly the same initial states and settings must
    # reproduce the numerical files byte-for-byte. No Python surrogate is used.
    with tempfile.TemporaryDirectory(prefix="gnw-repeat-", dir=output) as repeat_dir:
        repeat = Path(repeat_dir)
        execute(base_command + [str(repeat)], output / "logs" / "repeat_run.log", root)
        files = ("trajectories.csv", "refined_trajectories.csv", "probes.csv",
                 "edges.csv", "steady_states.csv", "truth/parameters.json", "truth/jacobian.csv")
        repeated = {name: sha256(output / name) == sha256(repeat / name) for name in files}
    observed = output / "observed"
    observed.mkdir(exist_ok=True)
    visible_columns = ("run_id", "initial_id", "target", "dose", "time", "G1", "G2", "G3", "G4")
    with (output / "trajectories.csv").open(newline="") as source_csv, \
         (observed / "trajectories.csv").open("w", newline="") as learner_csv:
        rows = csv.DictReader(source_csv)
        visible = csv.DictWriter(learner_csv, fieldnames=visible_columns)
        visible.writeheader()
        for row in rows:
            # Raw acceptance files include hidden efficiency for diagnostics;
            # the separate visible view does not reveal rho or actual strength.
            visible.writerow({name: row[name] for name in visible_columns})
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/tschaffter/genenetweaver",
        "source_commit": SOURCE_COMMIT, "jar_url": JAR_URL,
        "jar_sha256": sha256(jar), "jar_version": "3.1.2 Beta",
        "java_version": version.strip(), "python_version": sys.version,
        "platform": platform.platform(), "machine": platform.machine(),
        "adapter_sha256": sha256(source),
        "actual_native_calls": ["GeneNetwork.computeDxydt", "HillGene.computeMRnaProductionRate",
                                "PerturbationSingleGene.applyPerturbation", "Solver.step"],
        "model_translation": False, "clip_states": False, "normalization": False,
        "process_noise": False, "formal_dataset": False, "model_training": False,
        "compile_command": compile_command, "run_command": base_command + [str(output)],
        "repeat_identical": repeated,
        "artifact_sha256": {name: sha256(output / name) for name in files},
        "protocol_clock_scale": 8, "refined_clock_scale": 16,
        "native_relative_tolerance": 1e-10, "refined_relative_tolerance": 1e-12,
        "limitations": ["Four-gene deterministic RNA-only acceptance, not formal experiment",
                        "Continuous model-relative RNA abundance, not sequencing counts",
                        "Three initial states are not biological single-cell replicates"],
        "observed_export": "observed/trajectories.csv (no hidden efficiency, parameters or derivatives)",
    }
    (output / "truth" / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all(repeated.values()):
        raise RuntimeError("Native repeat did not reproduce all outputs exactly")
    if not args.skip_check:
        execute([sys.executable, str(Path(__file__).with_name("check_acceptance.py")),
                 "--output-dir", str(output)], output / "logs" / "checks.log", root)
    print(f"GNW acceptance artifacts: {output}")
    if args.skip_check:
        print("Checks were skipped; generation success is NOT acceptance success.")


if __name__ == "__main__":
    main()
