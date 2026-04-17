#!/usr/bin/env bash
# Warm-then-measure runner. Emits:
#   ELAPSED=<seconds>
#   ACC=<correct>/<total>=<ratio>
#   METRIC=<composite>   (acc_ratio*100000 - elapsed; higher is better — but TSV uses lower_is_better so we negate)
#   SCORE=<-metric>      (lower is better; primary for autoresearch)
set -u
LOG=${1:-autoresearch-rag-combo.log}
export RAG7_CACHE=${RAG7_CACHE:-1}
export RAG7_CACHE_DIR=${RAG7_CACHE_DIR:-$HOME/.cache/rag7}
# Sandbox workaround: use local uv cache if home cache blocked
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.uv-cache}
# Sandbox workaround: remote backend calls need httpx — unset SOCKS proxy ENV
unset ALL_PROXY all_proxy FTP_PROXY ftp_proxy 2>/dev/null || true

# Warm run (populate cache). Output discarded.
uv run python -m tests.eval_v2 > /dev/null 2>&1

# Measured run
START=$(python3 -c "import time; print(time.perf_counter())")
uv run python -m tests.eval_v2 > "$LOG" 2>&1
END=$(python3 -c "import time; print(time.perf_counter())")

python3 - "$LOG" "$START" "$END" <<'PY'
import re, sys
log_path, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
elapsed = end - start
try:
    txt = open(log_path).read()
except OSError:
    txt = ""
m = re.search(r"^Total=(\d+)/(\d+)=([0-9.]+)", txt, re.M)
if m:
    correct, total, ratio = int(m.group(1)), int(m.group(2)), float(m.group(3))
else:
    correct, total, ratio = 0, 0, 0.0
# Accuracy-first lex: score = -acc_ratio*100000 + elapsed  (lower is better)
score = -ratio * 100000 + elapsed
print(f"ELAPSED={elapsed:.2f}")
print(f"ACC={correct}/{total}={ratio:.4f}")
print(f"SCORE={score:.2f}")
PY
