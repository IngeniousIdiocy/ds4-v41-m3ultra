#!/usr/bin/env python3
"""Check local files against bench/v41-manifest.json.

Everything is optional. Point it at whatever you have and it reports on that,
so a reader with only a fixture set can confirm the fixtures, and a reader with
a finished run can confirm the whole run.

  # what always works, from a checkout alone: the shipped receipt directories,
  # and a check that no deleted artifact is still lying around
  bench/verify-v41-manifest.py

  # a fixture set and a model
  bench/verify-v41-manifest.py --fixtures /path/to/fixtures --model M.gguf

  # a finished run: every *.text in the directory is matched against the
  # reference digests, so an unexpected output is a failure, not a silence
  bench/verify-v41-manifest.py --run-dir v41-dspark-20260913T...

Exit status is 0 only if nothing checked came out wrong.
"""

import argparse
import hashlib
import json
import os
import sys

OK, BAD, SKIP = "ok", "FAIL", "--"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.rows = []
        self.failures = 0
        self.checked = 0

    def add(self, status, what, detail=""):
        self.rows.append((status, what, detail))
        if status == BAD:
            self.failures += 1
        if status != SKIP:
            self.checked += 1

    def digest(self, what, path, expected, expected_prefix=None):
        if not path or not os.path.isfile(path):
            self.add(SKIP, what, "not supplied")
            return None
        actual = sha256_file(path)
        if expected:
            if actual == expected:
                self.add(OK, what, actual[:16])
            else:
                self.add(BAD, what, f"got {actual[:16]}, want {expected[:16]}")
        elif expected_prefix:
            if actual.startswith(expected_prefix):
                self.add(OK, what, f"{actual[:16]} (prefix-bound only)")
            else:
                self.add(BAD, what, f"got {actual[:16]}, want prefix {expected_prefix}")
        else:
            self.add(SKIP, what, f"{actual[:16]} (no digest in manifest)")
        return actual

    def size(self, what, path, expected_bytes):
        if not path or not os.path.isfile(path):
            self.add(SKIP, what, "not supplied")
            return
        actual = os.path.getsize(path)
        if actual == expected_bytes:
            self.add(OK, what, f"{actual} bytes")
        else:
            self.add(BAD, what, f"{actual} bytes, want {expected_bytes}")

    def print(self):
        width = max((len(w) for _, w, _ in self.rows), default=10)
        for status, what, detail in self.rows:
            mark = {OK: "  ok  ", BAD: " FAIL ", SKIP: "  --  "}[status]
            print(f"[{mark}] {what:<{width}}  {detail}")
        print(f"\n{self.checked} checked, {self.failures} failed")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(here, "v41-manifest.json"))
    ap.add_argument("--repo-root", default=os.path.dirname(here))
    ap.add_argument("--fixtures", help="directory holding the fixture files")
    ap.add_argument("--model", help="target GGUF, checked by byte count only")
    ap.add_argument("--mtp-model", help="DSpark support GGUF, checked by byte count only")
    ap.add_argument("--bench-bin", help="ds4-bench to identify")
    ap.add_argument("--server-bin", help="ds4-server to identify")
    ap.add_argument("--references", help="directory holding upstream reference .text files")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="a finished run directory; every *.text in it is matched")
    args = ap.parse_args()

    with open(args.manifest) as f:
        m = json.load(f)
    r = Report()

    # Artifacts an earlier draft of this branch shipped and this one deleted.
    # A leftover file is harmless but misleading, so it is reported rather than
    # ignored: the binary does not read it any more.
    for stale in ("dspark/calib-ds41-code.txt",):
        if os.path.exists(os.path.join(args.repo_root, stale)):
            r.add(BAD, f"deleted artifact {stale}",
                  "still present in this checkout; the controller that read it is gone")
        else:
            r.add(OK, f"deleted artifact {stale}", "absent, as expected")

    # Receipt directories shipped in the repository carry their own hash file,
    # so this check needs nothing but a checkout.
    receipts_root = os.path.join(args.repo_root, "bench", "receipts")
    if os.path.isdir(receipts_root):
        for dirpath, _dirnames, filenames in os.walk(receipts_root):
            if "receipt-hashes.json" not in filenames:
                continue
            rel = os.path.relpath(dirpath, args.repo_root)
            with open(os.path.join(dirpath, "receipt-hashes.json")) as f:
                want = json.load(f)
            for name in sorted(want):
                r.digest(f"receipt {rel}/{name}",
                         os.path.join(dirpath, name), want[name])
            extra = sorted(set(filenames) - set(want) - {"receipt-hashes.json"})
            for name in extra:
                r.add(BAD, f"receipt {rel}/{name}", "present but not in receipt-hashes.json")

    if args.model:
        r.size("target weights", args.model, m["weights"]["target"]["bytes"])
    if args.mtp_model:
        r.size("dspark drafter", args.mtp_model, m["weights"]["dspark_drafter"]["bytes"])

    for label, path in (("ds4-bench", args.bench_bin), ("ds4-server", args.server_bin)):
        if not path:
            continue
        actual = sha256_file(path)
        for build, spec in m["binaries"].items():
            entry = spec.get(label)
            if not entry:
                continue
            want, prefix = entry.get("sha256"), entry.get("sha256_prefix")
            if (want and actual == want) or (prefix and actual.startswith(prefix)):
                bound = "exact" if want else "prefix-bound only"
                r.add(OK, f"{label} identity", f"{build} @ {spec['commit'][:12]} ({bound})")
                break
        else:
            r.add(BAD, f"{label} identity",
                  f"{actual[:16]} matches no build in the manifest")

    if args.fixtures:
        for name, spec in m["fixtures"].items():
            path = os.path.join(args.fixtures, name)
            if not os.path.isfile(path):
                r.add(SKIP, f"fixture {name}", "not present")
                continue
            r.digest(f"fixture {name}", path, spec["sha256"])

    refs = {spec["sha256"]: name for name, spec in m["references"].items()
            if isinstance(spec, dict) and "sha256" in spec}

    if args.references:
        for name, spec in m["references"].items():
            if not isinstance(spec, dict) or "sha256" not in spec:
                continue
            # Reference files are conventionally named for the binary that made
            # them, so accept both the bare name and the -upstream-<base> form.
            base = m["binaries"]["upstream"]["commit"][:7]
            for candidate in (f"{name}.text", f"{name}.txt",
                              f"{name}-upstream-{base}.text",
                              f"{name}-upstream-{base}.txt"):
                path = os.path.join(args.references, candidate)
                if os.path.isfile(path):
                    r.digest(f"reference {name}", path, spec["sha256"])
                    break
            else:
                r.add(SKIP, f"reference {name}", "not present")

    # A run directory: every emitted text must match some reference digest.
    # An output that matches nothing is a failure, not a silence.
    for run_dir in args.run_dir:
        if not os.path.isdir(run_dir):
            r.add(BAD, f"run {run_dir}", "not a directory")
            continue
        texts = sorted(f for f in os.listdir(run_dir) if f.endswith(".text"))
        if not texts:
            r.add(SKIP, f"run {os.path.basename(run_dir)}", "no .text files")
        for name in texts:
            path = os.path.join(run_dir, name)
            actual = sha256_file(path)
            label = f"{os.path.basename(run_dir)}/{name}"
            if actual in refs:
                r.add(OK, label, f"matches reference {refs[actual]}")
            elif os.path.getsize(path) == 0:
                r.add(SKIP, label, "empty")
            else:
                r.add(BAD, label, f"{actual[:16]} matches no reference digest")

    r.print()
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
