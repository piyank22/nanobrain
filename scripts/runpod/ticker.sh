#!/bin/bash
# Once a minute: write $W/status.json (phase, cost, data, every GPU, each run's state and last train line) and
# append a TICK line to $W/progress.log.
#   bash ticker.sh <pod start time, e.g. 2026-11-18T08:14:19Z> <usd_per_hour>
W=${W:-/workspace}; D=${D:-/root/data}
START=$(date -d "$1" +%s); RATE=${2:-4.43}
while true; do
  el=$(( $(date +%s)-START ))
  python3 - "$el" "$RATE" "$W" "$D" > $W/status.json.tmp <<'PY' && mv $W/status.json.tmp $W/status.json
import json, os, re, subprocess, sys, time, glob
el, rate, W, D = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""
gpus = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").replace(" ", "").splitlines()
runs = {}
for d in sorted(glob.glob(f"{W}/results/*/")):
    name = os.path.basename(d.rstrip("/"))
    st = open(d + "state").read().strip() if os.path.exists(d + "state") else ""
    tl = sh(f"grep -oE '\\[ *[0-9]+/[0-9]+\\].*' {d}train.log 2>/dev/null | tail -1 | cut -c1-140")
    if st or tl: runs[name] = {"state": st, "train": tl}
print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_min": el // 60,
    "phase": open(f"{W}/phase").read().strip() if os.path.exists(f"{W}/phase") else "",
    "data_gb": sh(f"du -sb {D} 2>/dev/null | awk '{{printf \"%.1f\", $1/1e9}}'"),
    "gpu_util_mem": gpus[0] if gpus else "", "gpus": gpus, "cost_usd": f"{el / 3600 * rate:.2f}", "runs": runs}))
PY
  echo "$(date -u +%FT%TZ) TICK $(cat $W/status.json)" >> $W/progress.log
  sleep 60
done
