#!/usr/bin/env bash
# One-shot setup on a fresh CUDA box (Vast.ai / RunPod / Lambda). Run from anywhere:
#   curl -sL https://raw.githubusercontent.com/GuerdonL/tribe/main/scripts/vast_setup.sh | bash
# then:
#   cd tribe && nohup python -m tribe.train --preset gpu --resume > m1.log 2>&1 &
#   tail -f m1.log
set -euo pipefail
cd "${HOME}"
if [ ! -d tribe ]; then git clone https://github.com/GuerdonL/tribe.git; fi
cd tribe && git pull -q
pip install -q -U pip
pip install -q -U "jax[cuda12]" optax flax numpy pytest
python - <<'PY'
import jax; print("jax", jax.__version__, "devices:", jax.devices())
assert jax.devices()[0].platform == "gpu", "JAX does not see the GPU"
PY
python -m pytest tests/test_jax_world.py tests/test_model.py -q
python scripts/m0_jax.py --bench
echo "ready: python -m tribe.train --preset gpu --resume"
