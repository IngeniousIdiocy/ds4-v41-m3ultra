#!/usr/bin/env bash
# Reproduce one serial-decode or cold-prefill arm from the V4.1 Flash M3 Ultra
# release matrix, through ds4-bench, in its own process.
#
# Run only with exclusive GPU access and one model process on the machine.
# Deliberately does not hash the 483 GiB model; it checks its byte count.
#
#   bench/reproduce-v41-native.sh --model M.gguf --fixture F.txt --depth 8k
#
# Depths: 8k (512 generated), 32k (256), 62k (128), 62k-prefill (0 generated,
# frontier logits dumped). Everything else is a flag with a sane default.
set -euo pipefail

BIN=./ds4-bench
MODEL=
FIXTURE=
DEPTH=8k
OUT=
GEN=
MODEL_BYTES=518596067328
FIXTURE_SHA=
SKIP_SIZE_CHECK=0

usage() {
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
    cat <<'EOF'

  --model PATH        target GGUF (required)
  --fixture PATH      prompt file (required)
  --depth NAME        8k | 32k | 62k | 62k-prefill   (default 8k)
  --bin PATH          ds4-bench executable (default ./ds4-bench)
  --gen N             override the generated-token count for this depth
  --out DIR           output directory (default v41-<depth>-<UTC timestamp>)
  --fixture-sha256 H  refuse to run unless the fixture hashes to H
  --model-bytes N     expected model size (default 518596067328)
  --skip-size-check   accept any model size
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --fixture) FIXTURE=$2; shift 2 ;;
        --depth) DEPTH=$2; shift 2 ;;
        --bin) BIN=$2; shift 2 ;;
        --gen) GEN=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --fixture-sha256) FIXTURE_SHA=$2; shift 2 ;;
        --model-bytes) MODEL_BYTES=$2; shift 2 ;;
        --skip-size-check) SKIP_SIZE_CHECK=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$MODEL" ] && [ -n "$FIXTURE" ] || { echo "--model and --fixture are required" >&2; exit 2; }

case "$DEPTH" in
    8k)          CTX=8192;  DEFAULT_GEN=512 ;;
    32k)         CTX=32768; DEFAULT_GEN=256 ;;
    62k)         CTX=62000; DEFAULT_GEN=128 ;;
    62k-prefill) CTX=62000; DEFAULT_GEN=0   ;;
    *) echo "--depth must be 8k, 32k, 62k or 62k-prefill" >&2; exit 2 ;;
esac
GEN=${GEN:-$DEFAULT_GEN}
ALLOC=$((CTX + 768))
OUT=${OUT:-v41-${DEPTH}-$(date -u +%Y%m%dT%H%M%SZ)}

[ -x "$BIN" ] || { echo "missing executable: $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }
[ -f "$FIXTURE" ] || { echo "missing fixture: $FIXTURE" >&2; exit 2; }

file_bytes() { if stat -f %z "$1" >/dev/null 2>&1; then stat -f %z "$1"; else stat -c %s "$1"; fi; }
sha256() { shasum -a 256 "$1" | awk '{print $1}'; }

ACTUAL_BYTES=$(file_bytes "$MODEL")
if [ "$SKIP_SIZE_CHECK" = 0 ] && [ "$ACTUAL_BYTES" != "$MODEL_BYTES" ]; then
    echo "model is $ACTUAL_BYTES bytes, expected $MODEL_BYTES" >&2
    echo "pass --model-bytes N or --skip-size-check if this is deliberate" >&2
    exit 2
fi

ACTUAL_FIXTURE_SHA=$(sha256 "$FIXTURE")
if [ -n "$FIXTURE_SHA" ] && [ "$ACTUAL_FIXTURE_SHA" != "$FIXTURE_SHA" ]; then
    echo "fixture digest $ACTUAL_FIXTURE_SHA, expected $FIXTURE_SHA" >&2
    exit 2
fi

mkdir -p "$OUT"

# Drop inherited tuning and diagnostic variables so the arm runs the build's own
# defaults. DS4_METAL_MODEL_UNTRACKED is set below and is not a tuning lever.
while IFS='=' read -r name _; do
    case "$name" in DS4_*|MTL_*) unset "$name" || true ;; esac
done < <(env)

{
    printf 'depth=%s\nctx=%s\ngen_tokens=%s\nctx_alloc=%s\n' "$DEPTH" "$CTX" "$GEN" "$ALLOC"
    printf 'model_bytes=%s\n' "$ACTUAL_BYTES"
    printf 'fixture=%s\nfixture_bytes=%s\nfixture_sha256=%s\n' \
        "$(basename "$FIXTURE")" "$(file_bytes "$FIXTURE")" "$ACTUAL_FIXTURE_SHA"
    printf 'bench_sha256=%s\n' "$(sha256 "$BIN")"
    git rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
    git status --porcelain 2>/dev/null | sed 's/^/dirty=/' || true
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$OUT/identity.txt"

ARGS=(-m "$MODEL" --metal --prompt-file "$FIXTURE"
      --ctx-start "$CTX" --ctx-max "$CTX" --ctx-alloc "$ALLOC"
      --gen-tokens "$GEN" --csv "$OUT/arm.csv")
if [ "$GEN" -gt 0 ]; then
    ARGS+=(--show-output)
else
    # --dump-frontier-logits-dir does not create its directory. It writes at the
    # end of prefill, which is why this depth generates nothing.
    mkdir -p "$OUT/logits"
    ARGS+=(--dump-frontier-logits-dir "$OUT/logits")
fi

set +e
DS4_METAL_MODEL_UNTRACKED=1 "$BIN" "${ARGS[@]}" >"$OUT/arm.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" >"$OUT/exit-code.txt"

# The decoded text is what identity is judged on. Extract it verbatim.
if [ "$GEN" -gt 0 ]; then
    python3 - "$OUT/arm.log" "$OUT/arm.text" <<'PY'
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r'ds4-bench: gen\[ctx=\d+\] decoded text: "', log)
if m:
    rest = log[m.end():]
    open(sys.argv[2], "w", encoding="utf-8").write(rest[:rest.rfind('"')])
PY
    if [ -s "$OUT/arm.text" ]; then
        printf 'text_bytes=%s\ntext_sha256=%s\n' \
            "$(file_bytes "$OUT/arm.text")" "$(sha256 "$OUT/arm.text")" >>"$OUT/identity.txt"
    fi
fi

echo "$OUT"
tail -3 "$OUT/arm.csv" 2>/dev/null || true
exit "$RC"
