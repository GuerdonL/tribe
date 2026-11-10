import base64
import importlib.util
import json
import os

import pytest

spec = importlib.util.spec_from_file_location("fleet", os.path.join(os.path.dirname(__file__), "..", "scripts", "fleet.py"))
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


def test_parse_jobs():
    jobs = fleet.parse_jobs("# c\n\na_1: python -m tribe.train --out runs/a_1 --lr 1e-4\nb: echo hi > runs/b/x\n")
    assert jobs == [("a_1", "python -m tribe.train --out runs/a_1 --lr 1e-4"), ("b", "echo hi > runs/b/x")]


@pytest.mark.parametrize("text", ["no colon here", "a: x runs/a\na: y runs/a", "a: python train.py --out runs/other"])
def test_parse_jobs_rejects(text):
    with pytest.raises(ValueError):
        fleet.parse_jobs(text)


def test_onstart_carries_the_job_and_pins_the_commit():
    job = dict(name="a", cmd="python x.py --out 'runs/a' && echo \"done\"", ref="abc123", hf_repo="u/r",
               on_done="destroy", sync_s=600, repo_url="https://github.com/u/tribe.git")
    cmd = fleet.onstart_cmd(job, "u/tribe")
    assert "raw.githubusercontent.com/u/tribe/abc123/scripts/worker.sh" in cmd
    b64 = cmd.split("TRIBE_JOB_B64=")[1].split(";")[0]
    assert json.loads(base64.b64decode(b64)) == job
    assert len(cmd) < 4000          # onstart commands have a size limit
