#!/bin/bash
# One unattended nanobrain session on a RunPod pod, run from a clone of this repo:
#   labless login check -> setup -> download -> runs one after another on GPU 0 (train, eval, submit) -> stop the pod.
#
#   MODE=smoke RUNS="smoke_t5:config.yaml" bash scripts/runpod/session.sh
#   MODE=full SUBMIT=1 RUNS="sincos-s0:config_sincos.yaml;sincosL-s0:config_sincosL.yaml" bash scripts/runpod/session.sh
#
# Runs are sequential on one GPU with 16 CPUs (the labless reference: 1 H100 + 16 CPUs), the least busy cores per run.
# Full sessions need $W on a network volume (marker file $W/.network-volume): run folders, the labless token and
# the logs live there, so they outlive the pod. Smoke runs are never submitted.
# Problems are written as "ISSUE:" lines to $W/progress.log and $W/issues.log (the monitor page shows them).
set -uo pipefail

MODE=${MODE:-smoke}; SUBMIT=${SUBMIT:-0}
HOLD_MIN=${HOLD_MIN:-15}                 # minutes to stay up after finishing, to collect results
NCPU=${NCPU:-16}
W=${W:-/workspace}; D=${D:-/root/data}; P=${P:-/root/pretrained/facebook_vit-mae-base}
N=$(cd "$(dirname "$0")/../.." && pwd)   # the repo this script lives in
RUNS=${RUNS:-smoke_t5:config.yaml}
LOG=$W/progress.log; R=$W/results
mkdir -p $R $D

export UV_CACHE_DIR=/root/.cache/uv HF_HOME=/root/.cache/huggingface PIP_CACHE_DIR=/root/.cache/pip
export BRAINMARKS_SMRI_DATA=$D HF_XET_HIGH_PERFORMANCE=1 PATH=/root/.local/bin:$PATH

log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
phase(){ echo "$1" > $W/phase; log "PHASE $1"; }
issue(){ log "ISSUE: $*"; echo "$(date -u +%FT%TZ) $*" >> $W/issues.log; }
stop_pod(){ log "STOPPING POD: $1"; sync; bash $N/scripts/runpod/stop.sh "$1"; }
abort(){ issue "$1"; phase "FAILED: $1"; stop_pod "aborted: $1"; exit 1; }

# --- 0. checks that need no setup -------------------------------------------------------------------------------
if [ "$MODE" != smoke ] && [ ! -f $W/.network-volume ]; then
  abort "full session needs $W on a network volume (marker $W/.network-volume missing)"
fi
IFS=';' read -ra RUNLIST <<< "$RUNS"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ $NGPU -ge 1 ] || abort "no GPU visible"

OUT=$N/output
if [ "$MODE" != smoke ]; then
  OUT=$W/output; mkdir -p $OUT $W/labless /root/.cache
  # keep a token from a login made before this script started: move it onto the volume, then link
  if [ -d /root/.cache/nanobrain ] && [ ! -L /root/.cache/nanobrain ]; then
    cp -n /root/.cache/nanobrain/labless_token.json $W/labless/ 2>/dev/null; rm -rf /root/.cache/nanobrain
  fi
  [ -L /root/.cache/nanobrain ] || ln -s $W/labless /root/.cache/nanobrain
fi

# Labless login: the cached token and the /api/auth/github/me call that scripts/submit.py makes before submitting.
labless_ok(){
  python3 - <<'PY' 2>/dev/null
import json, pathlib, sys, urllib.request
p = pathlib.Path.home() / ".cache" / "nanobrain" / "labless_token.json"
if not p.exists(): sys.exit(1)
tok = json.loads(p.read_text())["access_token"]
req = urllib.request.Request("https://api.labless.dev/api/auth/github/me",
                             headers={"Authorization": f"Bearer {tok}", "Accept": "application/json",
                                      "User-Agent": "nanobrain-submit/0.1"})
with urllib.request.urlopen(req, timeout=30) as r:
    print(json.loads(r.read())["login"])
PY
}
T0=$(date +%s)
if [ "$MODE" != smoke ] && [ "$SUBMIT" = 1 ]; then
  phase "login check"
  who=$(labless_ok) || abort "labless login missing or rejected: no run started. Fix: on a pod with this network volume, cd $N && uv run --no-sync python scripts/submit.py --login-only"
  log "labless login OK: $who"
fi
GIT="$(cd $N && git rev-parse --abbrev-ref HEAD) @ $(cd $N && git rev-parse --short HEAD)"
[ -z "$(cd $N && git status --porcelain)" ] || issue "repo is dirty before training: $(cd $N && git status --porcelain | head -3 | tr '\n' ' ')"
log "code: $(cd $N && git remote get-url origin) $GIT; runs: $RUNS; GPUs: $NGPU"

# --- 1. setup and data ------------------------------------------------------------------------------------------
phase setup
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh) >> $R/setup.log 2>&1 || abort "setup: uv install"
(cd $N && uv sync) >> $R/setup.log 2>&1 || abort "setup: uv sync"
(cd $N && uv run --no-sync hf download facebook/vit-mae-base model.safetensors config.json --local-dir $P) \
  >> $R/setup.log 2>&1 || abort "setup: ImageNet MAE-B weights"
log "setup done in $(( $(date +%s)-T0 ))s (torch $(cd $N && uv run --no-sync python -c 'import torch;print(torch.__version__)'))"

timed_dl(){ # name pattern...
  local name=$1; shift
  local inc=(); for p in "$@"; do inc+=(--include "$p"); done
  local b0=$(du -sb $D | cut -f1) t0=$(date +%s)
  (cd $N && uv run --no-sync hf download medarc/brainmarks-smri --repo-type dataset --local-dir $D "${inc[@]}") \
    >> $R/download_$name.log 2>&1 || return 1
  local b1=$(du -sb $D | cut -f1) s=$(( $(date +%s)-t0 )); [ $s -lt 1 ] && s=1
  log "download $name bytes=$((b1-b0)) seconds=$s MBps=$(( (b1-b0)/s/1000000 ))"
}
# Both modes first get the tables, 20 shards and the eval data: enough for a smoke run or the full-session preflight.
# Full sessions download the remaining shards after the preflight passes.
phase download
timed_dl tables "fomo300k/tables/*" || abort "download: tables"
timed_dl shards20 "fomo300k/shards/fomo300k-000*.zip" "fomo300k/shards/fomo300k-001*.zip" || abort "download: shards (first 20)"
timed_dl evaldata "abide1/*" "adhd200/*" "brats2021/*" "cnp/*" "ixi/*" "openbhb/*" "soop/*" "ucsf_pdgm/*" "upenn_gbm/*" \
  || abort "download: eval data"

make_filelist(){ # the QC training list restricted to the shards on disk -> /root/filelist.txt
  (cd $N && uv run --no-sync python -c "
import pandas as pd, pathlib
s = pd.read_parquet('$D/fomo300k/tables/shards.parquet', columns=['path','shard']).set_index('path')['shard']
have = {p.relative_to('$D/fomo300k').as_posix() for p in pathlib.Path('$D/fomo300k/shards').glob('*.zip')}
paths = [p for p in open('src/nanobrain/config/fomo300k_images_qc.txt').read().split() if s.get(p) in have]
open('/root/filelist.txt','w').write('\n'.join(paths)); print(len(paths), 'images in', len(have), 'shards')
") >> $LOG 2>&1
}
EXTRA=""
if [ "$MODE" = smoke ]; then
  make_filelist || abort "smoke file list"
  EXTRA="time_limit=10 train_filelist=/root/filelist.txt"
fi

# --- 2. runs: one after another on GPU 0 -------------------------------------------------------------------------
pick_cpus(){ # n -> the n least busy allowed cores over 2 s, comma separated
  python3 - "$1" <<'PY'
import os, sys, time
n = int(sys.argv[1])
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = list(map(int, f[1:])); out[int(f[0][3:])] = (v[3] + v[4], sum(v))
    return out
a = snap(); time.sleep(2); b = snap()
busy = {c: 1 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in b}
cands = sorted((c for c in os.sched_getaffinity(0) if c in busy), key=lambda c: busy[c])
if len(cands) < n: print(f"only {len(cands)} cores available for {n}", file=sys.stderr)
print(",".join(map(str, sorted(cands[:n]))))
PY
}
scores_line(){ # run_dir -> one line of scores
  python3 - "$1" <<'PY' 2>&1
import json, sys
s = json.load(open(sys.argv[1] + "/eval/aggregate.json"))["summary"]
keys = ["open", "segmentation", "global"] + sorted(k for k in s if k not in ("open", "segmentation", "global", "all"))
print(" ".join(f"{k}={s[k]['score']:.4f}" for k in keys if k in s))
PY
}
train_line(){ # run_dir -> steps etc. from summary.json
  python3 -c "
import json; d=json.load(open('$1/summary.json'))
print(' '.join(f'{k}={d[k]}' for k in ('step','epochs','wall_seconds','cpus','peak_memory_gb','final_loss') if k in d))" 2>&1
}
submit_link(){ # submit.log -> labless run link (submit.py prints "submitted <dir>: <json>")
  python3 - "$1" <<'PY' 2>/dev/null || echo "(see submit.log)"
import json, re, sys
line = [l for l in open(sys.argv[1]) if l.startswith("submitted ")][-1]
r = json.loads(line.split(": ", 1)[1])
def find(o):
    if isinstance(o, dict):
        for k in ("web_url", "url", "run_url"):
            if isinstance(o.get(k), str) and o[k].startswith("http"): return o[k]
        for k in ("run_id", "id"):
            if isinstance(o.get(k), str) and o[k].startswith("run_"): return "https://labless.dev/runs/" + o[k]
        for v in o.values():
            u = find(v)
            if u: return u
    return None
print(find(r) or line.strip()[:200])
PY
}
run_one(){ # name config cpus
  local name=$1 cfg=$2 cpus=$3 RR=$R/$1; mkdir -p $RR
  echo "train" > $RR/state; log "RUN $name: train on GPU 0, CPUs $cpus, $cfg"
  (cd $N && CUDA_VISIBLE_DEVICES=0 taskset -c $cpus uv run --no-sync python -m nanobrain.train --config $cfg \
    --overrides output_dir=$OUT natinit.path=$P name=$name $EXTRA) > $RR/train.log 2>&1 \
    || { echo "failed" > $RR/state; issue "RUN $name: training failed (tail: $(tail -2 $RR/train.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
  log "RUN $name TRAINSUM $(train_line $OUT/$name)"
  echo "eval" > $RR/state
  (cd $N && CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m brainmarks_smri vit_mae_3d open --overrides \
    "imports=[nanobrain.model]" model_kwargs.checkpoint=$OUT/$name/checkpoint-last.pth output_dir=$OUT/$name name=eval) \
    > $RR/eval.log 2>&1 \
    || { echo "failed" > $RR/state; issue "RUN $name: eval failed (tail: $(tail -2 $RR/eval.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
  log "RUN $name SCORES $(scores_line $OUT/$name)"
  cp $OUT/$name/summary.json $OUT/$name/config.yaml $RR/ 2>/dev/null; cp -r $OUT/$name/eval $RR/ 2>/dev/null
  if [ "$MODE" != smoke ] && [ "$SUBMIT" = 1 ] && [[ "$name" != smoke* ]]; then
    echo "submit" > $RR/state
    if (cd $N && uv run --no-sync python scripts/submit.py $OUT/$name) > $RR/submit.log 2>&1; then
      log "RUN $name SUBMITTED $(submit_link $RR/submit.log)"
    else
      issue "RUN $name: submit failed (results are safe in $OUT/$name). Resubmit: cd $N && uv run --no-sync python scripts/submit.py $OUT/$name. $(tail -1 $RR/submit.log | cut -c1-200)"
    fi
  fi
  echo "done" > $RR/state
}

# --- preflight (full sessions): every planned config must train, load in the eval and pass the submission
# checks before the big download and the hour-long runs. ~5 min per config; any failure stops the pod.
preflight(){
  phase preflight
  make_filelist || abort "preflight: file list"
  local PF=/root/preflight spec name cfg pn cpus bad
  for spec in "${RUNLIST[@]}"; do
    name=${spec%%:*}; cfg=${spec#*:}; pn=pf-$name
    [ -f $N/$cfg ] || continue                      # reported when the runs start
    cpus=$(pick_cpus $NCPU 2>>$LOG)
    (cd $N && CUDA_VISIBLE_DEVICES=0 taskset -c $cpus uv run --no-sync python -m nanobrain.train --config $cfg \
      --overrides output_dir=$PF natinit.path=$P name=$pn time_limit=3 train_filelist=/root/filelist.txt) \
      > $R/preflight_$name.train.log 2>&1 \
      || abort "preflight $name: training failed (tail: $(tail -2 $R/preflight_$name.train.log | tr '\n' ' ' | cut -c1-300))"
    (cd $N && CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m brainmarks_smri vit_mae_3d ixi_age --overrides \
      "imports=[nanobrain.model]" model_kwargs.checkpoint=$PF/$pn/checkpoint-last.pth output_dir=$PF/$pn name=eval) \
      > $R/preflight_$name.eval.log 2>&1 \
      || abort "preflight $name: eval could not load or score the checkpoint (tail: $(tail -2 $R/preflight_$name.eval.log | tr '\n' ' ' | cut -c1-300))"
    # a one-task eval has no open score, so the dry run must report exactly that and nothing else
    (cd $N && uv run --no-sync python scripts/submit.py --dry-run $PF/$pn) > $R/preflight_$name.submit.log 2>&1
    bad=$(grep '^error:' $R/preflight_$name.submit.log | grep -v 'eval is incomplete')
    [ -z "$bad" ] || abort "preflight $name: submission check failed: $(echo $bad | cut -c1-300)"
    log "PREFLIGHT $name OK: $(train_line $PF/$pn); eval loaded and scored ixi_age; submission checks passed"
  done
}
if [ "$MODE" != smoke ]; then
  preflight
  phase download
  timed_dl shards "fomo300k/shards/*" || abort "download: remaining shards"
fi

if [ "$MODE" != smoke ] && [ "$SUBMIT" = 1 ]; then
  who=$(labless_ok) || abort "labless login stopped working before training: no run started"
fi
phase runs
log "running ${#RUNLIST[@]} run(s) one after another on GPU 0"
fails=0
run_ended_badly(){ # name: count it, and flag it if the run died before writing its own issue
  fails=$((fails+1))
  grep -q "RUN $1:" $W/issues.log 2>/dev/null || issue "RUN $1: ended with an error (see $R/$1/)"
  echo "failed" > $R/$1/state 2>/dev/null
}
for spec in "${RUNLIST[@]}"; do
  name=${spec%%:*}; cfg=${spec#*:}
  [ -f $N/$cfg ] || { issue "RUN $name: config $cfg not found, skipped"; continue; }
  cpus=$(pick_cpus $NCPU 2>>$LOG)         # fresh pick per run: the least busy cores right now
  [ -n "$cpus" ] || { issue "RUN $name: no CPUs could be picked, skipped"; fails=$((fails+1)); continue; }
  run_one "$name" "$cfg" "$cpus" || run_ended_badly "$name"
done

if [ -s $W/issues.log ]; then phase "done with issues"; else phase done; fi
log "ALL DONE in $(( $(date +%s)-T0 ))s, $fails failed run(s); holding $HOLD_MIN min (touch $W/cancel_stop to keep the pod up)"
for i in $(seq $((HOLD_MIN*6))); do [ -f $W/cancel_stop ] && { log "stop cancelled"; rm -f $W/cancel_stop; exit 0; }; sleep 10; done
stop_pod "all steps done"
