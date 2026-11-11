"""Inside a box-boy job: upload one file of ANOTHER run to that run's folder in the results repo.
usage: python scripts/put_result.py <run> <file in runs/<run>/>
The job's own folder (runs/<job name>/) is uploaded by box-boy; this is for a job that adds a result to a run
it pulled with bb-pull, such as an evaluation writing runs/<run>/snap_eval.csv. Needs BB_HF_REPO and HF_TOKEN,
which box-boy sets on the instance."""
import os
import sys

from huggingface_hub import HfApi

run, name = sys.argv[1], sys.argv[2]
repo = os.environ["BB_HF_REPO"]
HfApi().upload_file(path_or_fileobj=os.path.join("runs", run, name), path_in_repo=f"{run}/{name}", repo_id=repo,
                    commit_message=f"{run}: {name}")
print(f"[put_result] {run}/{name} -> {repo}")
