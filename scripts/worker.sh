#!/usr/bin/env bash
# Runs ONE job on a fresh GPU instance. Started by the instance's onstart command, which
# scripts/fleet.py writes; not meant to be run by hand.
#
# Reads: TRIBE_JOB_B64 (base64 JSON: name, cmd, ref, hf_repo, on_done, sync_s), HF_TOKEN,
#        and Vast's CONTAINER_ID / CONTAINER_API_KEY.
# Does:  clone the repo at the given commit, install pinned JAX, restore any earlier state of
#        this run from the Hugging Face repo, run the command, upload runs/<name> while it runs
#        and when it ends, then destroy this instance (or stop it if the command failed).
set -uo pipefail
LOG=/root/tribe_worker.log
exec > >(tee -a "$LOG") 2>&1
echo "[worker] start $(date -u +%FT%TZ)"

job() { python3 -c "import base64,json,os,sys; print(json.loads(base64.b64decode(os.environ['TRIBE_JOB_B64']))[sys.argv[1]])" "$1"; }
NAME=$(job name); CMD=$(job cmd); REF=$(job ref); HF_REPO=$(job hf_repo); ON_DONE=$(job on_done); SYNC_S=$(job sync_s)
REPO_URL=$(job repo_url)
export HF_TOKEN XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "[worker] job=$NAME ref=$REF hf=$HF_REPO on_done=$ON_DONE"

finish() {   # $1 = exit code of the job
    python scripts/fleet_sync.py push "$HF_REPO" "$NAME" --all --status "$1" --log "$LOG" || echo "[worker] final upload FAILED"
    if [ "$1" = "0" ] && [ "$ON_DONE" = "destroy" ]; then
        echo "[worker] done, destroying instance $CONTAINER_ID"
        vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" \
            || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    elif [ "$ON_DONE" != "keep" ]; then
        echo "[worker] stopping instance $CONTAINER_ID (exit code $1; disk kept for inspection)"
        vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    fi
}

[ -f /etc/environment ] && set -a && . /etc/environment && set +a
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
cd /root
if [ ! -d tribe/.git ]; then git clone -q "$REPO_URL" tribe || { echo "[worker] clone failed"; exit 1; }; fi
cd tribe && git fetch -q origin && git checkout -q "$REF" || { echo "[worker] checkout $REF failed"; exit 1; }
PIP="pip install -q"; command -v uv >/dev/null && PIP="uv pip install -q"
$PIP -r requirements-gpu.txt || { echo "[worker] install failed"; finish 97; exit 97; }
python -c "import jax; d=jax.devices(); print('[worker] jax', jax.__version__, d); assert d[0].platform=='gpu'" \
    || { echo "[worker] JAX does not see the GPU"; finish 98; exit 98; }

# an earlier attempt at this run (interrupted instance, or a rerun): restore it, or stop if it already finished
python scripts/fleet_sync.py pull "$HF_REPO" "$NAME"; rc=$?
if [ $rc = 3 ]; then echo "[worker] run already finished"; finish 0; exit 0; fi

( while sleep "$SYNC_S"; do python scripts/fleet_sync.py push "$HF_REPO" "$NAME" --log "$LOG"; done ) &
SYNC_PID=$!
echo "[worker] running: $CMD"
bash -c "$CMD"; rc=$?
kill $SYNC_PID 2>/dev/null
echo "[worker] command exited with $rc at $(date -u +%FT%TZ)"
finish $rc
