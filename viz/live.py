"""Live training logs, read straight from the running instances.

The web server starts one `Live` poller. Once a minute it lists this project's Vast.ai instances (the ones box-boy
labelled `<project>:<job>`); every few seconds it fetches, over SSH, the bytes of each instance's
`runs/<job>/log.csv` that it does not have yet. Nothing waits for the 10-minute upload to Hugging Face.

It needs what `bb logs` needs: the `vastai` command (logged in) and this machine's SSH key on the instances,
which `bb launch` attaches. Standard library only. A run that ends stays listed for a few hours.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE_FILE = os.path.join(HERE, "data", "live.json")
LIST_EVERY_S = 60
POLL_EVERY_S = 8
KEEP_ENDED_S = 6 * 3600
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
            "-o", "ControlPath=~/.ssh/tribe-viz-%C", "-o", "ControlPersist=120"]


def find_vastai():
    for c in (shutil.which("vastai"), os.path.expanduser("~/.local/bin/vastai"),
              os.path.expanduser("~/box-boy/.venv/bin/vastai")):
        if c and os.path.exists(c):
            return c
    return None


def project_name():
    try:
        cfg = tomllib.load(open(os.path.join(ROOT, "boxboy.toml"), "rb"))
        if cfg.get("name"):
            return cfg["name"], cfg.get("runs_dir", "runs")
    except (OSError, ValueError):
        pass
    return os.path.basename(ROOT), "runs"


def planned_updates():
    """{job name: --updates N} from the jobs files, so progress has a denominator."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, "jobs", "*.txt")):
        try:
            for line in open(path):
                m = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.+)$", line.strip())
                if m and not line.lstrip().startswith("#"):
                    u = re.search(r"--updates\s+(\d+)", m.group(2))
                    out[m.group(1)] = int(u.group(1)) if u else 3000        # the gpu_xl preset's default
        except OSError:
            pass
    return out


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


class Run:
    def __init__(self, name):
        self.name = name
        self.instance = None
        self.state = "starting"          # starting | loading | setup | training | ended
        self.detail = ""
        self.gpu = ""
        self.dph = None
        self.total = None
        self.header = None
        self.rows = []                   # list of lists of numbers, in header order
        self.offset = 0                  # bytes of log.csv consumed (whole lines only)
        self.ssh = None
        self.ssh_fails = 0
        self.hf_t = 0                    # when the uploaded copy was last fetched
        self.first_seen = time.time()
        self.last_row_t = None
        self.ended_t = None

    def public(self, start):
        cols = {}
        if self.header:
            for j, k in enumerate(self.header):
                cols[k] = [r[j] for r in self.rows[start:]]
        return dict(name=self.name, instance=self.instance, state=self.state, detail=self.detail, gpu=self.gpu,
                    dph=self.dph, total=self.total, n=len(self.rows), start=start, cols=cols,
                    last_row_age_s=None if self.last_row_t is None else round(time.time() - self.last_row_t, 1),
                    ended_age_s=None if self.ended_t is None else round(time.time() - self.ended_t))


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.runs = {}
        self.vastai = find_vastai()
        self.project, self.runs_dir = project_name()
        self.error = None if self.vastai else "the vastai command was not found"
        self.updated = 0
        self.hf_repo = None
        try:
            self.hf_repo = tomllib.load(open(os.path.join(ROOT, "boxboy.toml"), "rb")).get("hf_repo")
        except (OSError, ValueError):
            pass
        self._load()

    # ------------------------------------------------------------ state ---
    def _load(self):
        try:
            saved = json.load(open(STATE_FILE))
        except (OSError, ValueError):
            return
        for d in saved.get("runs", []):
            r = Run(d["name"])
            r.__dict__.update({k: d[k] for k in ("instance", "state", "detail", "gpu", "dph", "total", "header", "rows",
                                                 "offset", "first_seen", "last_row_t", "ended_t") if k in d})
            self.runs[r.name] = r

    def _save(self):
        with self.lock:
            data = dict(runs=[{k: v for k, v in r.__dict__.items() if k not in ("ssh", "ssh_fails", "hf_t")} for r in self.runs.values()])
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, STATE_FILE)

    def state(self, have):
        """What the page asks for. `have` is {run name: rows it already has}."""
        with self.lock:
            runs = [r.public(min(max(int(have.get(r.name, 0)), 0), len(r.rows))) for r in sorted(self.runs.values(), key=lambda r: r.name)]
            return dict(t=time.time(), error=self.error, project=self.project, poll_s=POLL_EVERY_S, runs=runs)

    # ---------------------------------------------------------- polling ---
    def _sh(self, args, timeout):
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            return r.returncode, r.stdout
        except (subprocess.TimeoutExpired, OSError):
            return 1, b""

    def _list(self):
        rc, out = self._sh([self.vastai, "show", "instances", "--raw"], 40)
        if rc != 0:
            self.error = "could not list Vast.ai instances"
            return
        try:
            rows = json.loads(out)
        except ValueError:
            self.error = "could not read the Vast.ai instance list"
            return
        self.error = None
        prefix = self.project + ":"
        planned = planned_updates()
        seen = set()
        with self.lock:
            for i in rows:
                label = i.get("label") or ""
                if not label.startswith(prefix):
                    continue
                name = label[len(prefix):]
                seen.add(name)
                r = self.runs.get(name)
                if r is None or (r.instance not in (None, i["id"]) and r.state == "ended"):
                    r = self.runs[name] = Run(name)            # a new job, or a relaunch of an ended one
                if r.instance != i["id"]:
                    r.instance, r.ssh = i["id"], None
                r.gpu = f"{i.get('num_gpus') or 1}x {i.get('gpu_name') or '?'}"
                r.dph = num(i.get("dph_total"))
                r.total = planned.get(name, r.total)
                r.ended_t = None
                actual = i.get("actual_status")
                if i.get("intended_status") == "stopped" or actual in ("exited", "stopped", "offline"):
                    r.state, r.detail = "ended", "instance stopped: the job failed or hit its time limit (bb logs, then bb destroy)"
                    r.ended_t = r.ended_t or time.time()
                elif actual != "running":
                    r.state, r.detail = "loading", "the instance is pulling its image"
                elif r.state in ("starting", "loading", "ended"):
                    r.state, r.detail = "setup", "installing dependencies and compiling"
            for name, r in self.runs.items():
                if name not in seen and r.state != "ended":
                    r.state, r.ended_t = "ended", time.time()
                    done = r.total and len(r.rows) >= r.total
                    r.detail = "finished; the instance destroyed itself" if done else "the instance is gone"
            for name in [n for n, r in self.runs.items() if r.ended_t and time.time() - r.ended_t > KEEP_ENDED_S]:
                del self.runs[name]

    def _fetch(self, r):
        if r.ssh is None:
            rc, out = self._sh([self.vastai, "ssh-url", str(r.instance)], 30)
            m = re.search(r"ssh://(\S+?)@(\S+?):(\d+)", out.decode(errors="replace"))
            if not m:
                return
            r.ssh = ["ssh"] + SSH_OPTS + ["-p", m.group(3), f"{m.group(1)}@{m.group(2)}"]
        path = f"/root/{self.project}/{self.runs_dir}/{r.name}/log.csv"
        rc, out = self._sh(r.ssh + [f"tail -c +{r.offset + 1} {path} 2>/dev/null"], 25)
        if rc == 255:                                          # no connection, or the key was refused
            r.ssh, r.ssh_fails = None, r.ssh_fails + 1         # the address may have changed; ask again next time
            if r.ssh_fails >= 2:
                self._fetch_uploaded(r)
            return
        r.ssh_fails = 0
        if rc != 0:
            return
        end = out.rfind(b"\n")
        if end < 0:
            return
        lines = out[: end + 1].decode(errors="replace").splitlines()
        with self.lock:
            r.offset += end + 1
            for line in lines:
                cells = line.strip().split(",")
                if r.header is None:
                    r.header = cells
                elif len(cells) == len(r.header):
                    r.rows.append([num(c) for c in cells])
                    r.last_row_t = time.time()
            if r.rows:
                r.state, r.detail = "training", ""
        self.updated = time.time()

    def _fetch_uploaded(self, r):
        """Fallback when SSH is refused: the copy of log.csv the instance uploads to the results repo."""
        if time.time() - r.hf_t < 60:
            return
        r.hf_t = time.time()
        try:
            from huggingface_hub import HfApi, hf_hub_download
            if self.hf_repo is None:
                self.hf_repo = f"{HfApi().whoami()['name']}/{self.project}-runs"
            text = open(hf_hub_download(self.hf_repo, f"{r.name}/log.csv", force_download=True)).read()
        except Exception:
            with self.lock:
                r.detail = "no SSH access to this instance, and nothing uploaded yet"
            return
        lines = text[: text.rfind("\n") + 1].splitlines()
        if len(lines) < 2:
            return
        header = lines[0].strip().split(",")
        rows = [[num(c) for c in ln.strip().split(",")] for ln in lines[1:] if ln.count(",") == len(header) - 1]
        with self.lock:
            if len(rows) > len(r.rows):
                r.header, r.rows, r.last_row_t = header, rows, time.time()
            if r.rows:
                r.state = "training"
            r.detail = "no SSH access to this instance: showing its uploads, up to 10 minutes behind"
        self.updated = time.time()

    def loop(self):
        if not self.vastai:
            return
        last_list = 0
        while True:
            try:
                if time.time() - last_list > LIST_EVERY_S:
                    self._list()
                    last_list = time.time()
                with self.lock:
                    active = [r for r in self.runs.values() if r.state in ("setup", "training") and r.instance]
                threads = [threading.Thread(target=self._fetch, args=(r,), daemon=True) for r in active]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(40)
                self._save()
            except Exception as e:                             # never let the poller die
                self.error = f"live poller: {type(e).__name__}: {e}"
            time.sleep(POLL_EVERY_S if any(r.state != "ended" for r in self.runs.values()) else 20)

    def start(self):
        threading.Thread(target=self.loop, daemon=True, name="live").start()
        return self
