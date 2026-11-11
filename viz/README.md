# viz: watch the trained models

A website for the training runs: every metric over training and across models, and a top-down player for
episodes of any saved model.

Runs store only metrics and weights, not episodes. So `build.py` plays each saved model (every `snap/uNNNNN.pkl`
and each run's latest `params.pkl` in the results repo) on the same 32 worlds, on the CPU, and records every step.
Because all models play the same worlds, different models can be drawn on one map.

```
viz/build.py     sync with GuerdonL/tribe-runs, record what is new, write viz/data/ (git-ignored)
viz/serve.py     web server: viz/site + viz/data, HTTP Basic auth, standard library only
viz/live.py      inside the server: reads each running instance's log.csv over SSH every few seconds (the Live panel)
viz/site/        the page (plain HTML, CSS and JavaScript; no build step)
viz/install.sh   systemd user services: the server, and a timer that runs build.py every 30 minutes
```

## Set up

The pinned JAX needs Python 3.12; `uv` fetches one without sudo.

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python jax==0.12.0.1 flax==0.12.12 optax==0.2.8 numpy==2.5.4 huggingface_hub
.venv/bin/python viz/build.py          # about 20 s per model, 700 MB of RAM; downloads the weights into runs/
viz/install.sh 8080                    # or by hand: .venv/bin/python viz/serve.py --port 8080
```

The login is one line, `user:password`, in `~/.config/tribe-viz/auth`, created with a random password on first
start. The connection is plain HTTP: the password keeps casual visitors out, it is not protected on the network path.

## Notes

- A recording is one sampled episode per world (actions are sampled, as in training), with fixed seeds: the same
  model always gives the same 32 episodes. Per-model numbers from 32 episodes are noisy; the training-log numbers
  beside them are means over the 10 updates before the snapshot.
- Changing `N_WORLDS`, the seeds or the recorded fields in `build.py` needs `VERSION` bumped (re-records everything).
- A run's latest weights (`params.pkl`) are re-recorded whenever the file changes, so a run in progress updates.

## Live runs

While box-boy instances are running, the page shows a Live panel: progress, time left, cost, and the training
metrics as they are logged. `live.py` lists the project's Vast.ai instances once a minute and fetches the new bytes
of each `runs/<job>/log.csv` over SSH every 8 seconds; the page asks `/live/state.json` for rows it does not have.
It needs what `bb logs` needs (the `vastai` login and this machine's SSH key on the instances). If an instance
refuses the key, it falls back to the copy uploaded to the results repo, up to 10 minutes behind, and says so.
After changing `serve.py` or `live.py`: `systemctl --user restart tribe-viz`.
