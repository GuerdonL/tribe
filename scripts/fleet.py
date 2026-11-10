"""Run tribe jobs on rented Vast.ai GPU instances, one job per instance.

    python scripts/fleet.py launch jobs/seeds.txt --gpu RTX_4090 --max-price 0.60        # asks before renting
    python scripts/fleet.py status
    python scripts/fleet.py logs <name>
    python scripts/fleet.py collect [name ...]      # download finished or in-progress runs into runs/
    python scripts/fleet.py destroy [name ...|--all]

A jobs file has one job per line, `name: command`. The command runs from the repo root on the
instance and must write to runs/<name>/ (that folder is what gets uploaded and restored):

    m1_s3: python -m tribe.train --preset gpu_xl --seed 3 --out runs/m1_s3 --resume
    m1_s4: python -m tribe.train --preset gpu_xl --seed 4 --out runs/m1_s4 --resume

Each instance clones the repo at the commit you launch from (it must be pushed), installs the pinned
requirements, runs its command, uploads runs/<name>/ to a private Hugging Face repo every few minutes
and at the end, then destroys itself. A failed job stops its instance instead (disk kept, GPU billing
off) so the log can be read. Needs: the `vastai` CLI logged in with an account key, and a Hugging Face
login on this machine (the token is passed to the instances as an environment variable).
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys

IMAGE = "vastai/pytorch:cuda-13.2.1-auto"
LABEL = "tribe:"
RAW = "https://raw.githubusercontent.com/{slug}/{ref}/scripts/worker.sh"


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(args[:4])} ...\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def vast(*args, check=True):
    out = sh("vastai", *args, "--raw", check=check)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def parse_jobs(text):
    """`name: command` per line; blank lines and # comments ignored."""
    jobs = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.+)$", line)
        if not m:
            raise ValueError(f"line {n}: expected `name: command`, got {line!r}")
        jobs.append((m.group(1), m.group(2)))
    names = [j[0] for j in jobs]
    dup = {x for x in names if names.count(x) > 1}
    if dup:
        raise ValueError(f"duplicate job names: {sorted(dup)}")
    for name, cmd in jobs:
        if f"runs/{name}" not in cmd:
            raise ValueError(f"job {name}: the command must write to runs/{name} (that folder is what gets saved)")
    return jobs


def onstart_cmd(job, slug):
    """The instance's onstart command: hand the job to scripts/worker.sh at the launch commit."""
    b64 = base64.b64encode(json.dumps(job).encode()).decode()
    url = RAW.format(slug=slug, ref=job["ref"])
    return f"export TRIBE_JOB_B64={b64}; curl -fsSL {url} -o /root/worker.sh && nohup bash /root/worker.sh > /dev/null 2>&1 &"


def offer_query(a):
    q = (f"gpu_name={a.gpu} num_gpus={a.num_gpus} rentable=true reliability>{a.reliability} inet_down>200 "
         f"disk_space>={a.disk} dph_total<={a.max_price} geolocation notin [{a.exclude_geo}]")
    return q


def git_ref_and_slug():
    ref = sh("git", "rev-parse", "HEAD")
    if not sh("git", "branch", "-r", "--contains", ref):
        sys.exit(f"commit {ref[:8]} is not on the remote; push it first (instances clone from GitHub)")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("warning: uncommitted changes will NOT be on the instances:\n" + dirty)
    url = sh("git", "remote", "get-url", "origin")
    slug = re.sub(r"^(git@github\.com:|https://github\.com/)", "", url).removesuffix(".git")
    return ref, slug


def hf_token_and_repo(repo):
    from huggingface_hub import HfApi, get_token
    tok = get_token()
    if not tok:
        sys.exit("no Hugging Face login found; run `hf auth login` first")
    return tok, repo or f"{HfApi().whoami()['name']}/tribe-runs"


def instances():
    return [i for i in vast("show", "instances") if (i.get("label") or "").startswith(LABEL)]


def cmd_launch(a):
    jobs = parse_jobs(open(a.jobs).read())
    ref, slug = git_ref_and_slug()
    token, hf_repo = hf_token_and_repo(a.hf_repo)
    running = {i["label"][len(LABEL):] for i in instances()}
    clash = [n for n, _ in jobs if n in running]
    if clash:
        sys.exit(f"already have instances for: {clash} (destroy them or rename the jobs)")
    # the host driver must support CUDA 12 (filtered here: the server-side cuda_max_good filter drops valid offers)
    offers = sorted((o for o in vast("search", "offers", offer_query(a), "-o", "dph_total+")
                     if (o.get("cuda_max_good") or 0) >= 12.0), key=lambda o: o["dph_total"])
    if len(offers) < len(jobs):
        sys.exit(f"only {len(offers)} offers match for {len(jobs)} jobs; raise --max-price or relax the filters")
    picks = offers[: len(jobs)]
    total = sum(o["dph_total"] for o in picks)
    print(f"commit {ref[:8]} of {slug}; results to hf:{hf_repo}; image {IMAGE}")
    for (name, cmd), o in zip(jobs, picks):
        print(f"  {name:20s} {o['num_gpus']}x {o['gpu_name']} ${o['dph_total']:.3f}/h {o.get('geolocation','?'):18s} | {cmd}")
    print(f"{len(jobs)} instances, ${total:.2f}/h in total while they run. Each destroys itself when its job succeeds.")
    if a.dry_run:
        return
    if not a.yes and input("rent these? [y/N] ").strip().lower() != "y":
        sys.exit("nothing rented")
    spare = offers[len(jobs):]
    for (name, cmd), o in zip(jobs, picks):
        job = dict(name=name, cmd=cmd, ref=ref, hf_repo=hf_repo, on_done=a.on_done, sync_s=a.sync_s,
                   repo_url=f"https://github.com/{slug}.git")
        while True:
            r = vast("create", "instance", str(o["id"]), "--image", IMAGE, "--disk", str(a.disk), "--label", LABEL + name,
                     "--ssh", "--direct", "--cancel-unavail", "--env", f"-e HF_TOKEN={token}",
                     "--onstart-cmd", onstart_cmd(job, slug), check=False)
            if isinstance(r, dict) and r.get("success"):
                print(f"  {name}: instance {r.get('new_contract')} on offer {o['id']}")
                break
            if not spare:
                print(f"  {name}: FAILED to rent ({r}); no offers left")
                break
            print(f"  {name}: offer {o['id']} was not available, trying the next one")
            o = spare.pop(0)


def cmd_status(a):
    rows = instances()
    if not rows:
        print("no tribe instances")
    for i in rows:
        up = (i.get("duration") or 0) / 3600
        print(f"{i['label'][len(LABEL):]:20s} id {i['id']}  {i.get('actual_status') or i.get('cur_state'):9s} "
              f"{i.get('num_gpus')}x {i.get('gpu_name')}  ${i.get('dph_total', 0):.3f}/h  up {up:.1f} h")
    if rows:
        print(f"total ${sum(i.get('dph_total', 0) for i in rows if i.get('actual_status') == 'running'):.2f}/h running")


def by_name(names):
    rows = instances()
    if names:
        rows = [i for i in rows if i["label"][len(LABEL):] in names]
        missing = set(names) - {i["label"][len(LABEL):] for i in rows}
        if missing:
            sys.exit(f"no instance for: {sorted(missing)}")
    return rows


def cmd_logs(a):
    for i in by_name([a.name]):
        print(sh("vastai", "logs", str(i["id"]), "--tail", str(a.tail), check=False))


def cmd_collect(a):
    from huggingface_hub import snapshot_download
    _, hf_repo = hf_token_and_repo(a.hf_repo)
    pats = [f"{n}/*" for n in a.names] or None
    path = snapshot_download(hf_repo, allow_patterns=pats, local_dir=a.runs)
    for d in sorted(os.listdir(path)):
        st = os.path.join(path, d, "STATUS.json")
        if os.path.isdir(os.path.join(path, d)) and (not a.names or d in a.names):
            state = f"exit code {json.load(open(st))['exit_code']}" if os.path.exists(st) else "in progress or not started"
            print(f"  {a.runs}/{d}: {state}")


def cmd_destroy(a):
    if not a.names and not a.all:
        sys.exit("name the jobs to destroy, or pass --all")
    rows = by_name(a.names)
    for i in rows:
        print(f"  {i['label'][len(LABEL):]} (id {i['id']}, {i.get('actual_status')})")
    if rows and (a.yes or input("destroy these instances and their disks? [y/N] ").strip().lower() == "y"):
        for i in rows:
            vast("destroy", "instance", str(i["id"]))
        print("destroyed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("launch")
    p.add_argument("jobs")
    p.add_argument("--gpu", default="RTX_4090")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-price", type=float, default=0.60, help="$/h per instance")
    p.add_argument("--disk", type=int, default=30, help="GB")
    p.add_argument("--reliability", type=float, default=0.98)
    p.add_argument("--exclude-geo", default="CN", help="comma-separated country codes (Hugging Face is unreachable from some)")
    p.add_argument("--on-done", choices=["destroy", "stop", "keep"], default="destroy")
    p.add_argument("--sync-s", type=int, default=600, help="seconds between uploads")
    p.add_argument("--hf-repo", default=None, help="default <your hf user>/tribe-runs")
    p.add_argument("--dry-run", action="store_true", help="show what would be rented and the cost, rent nothing")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_launch)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("logs"); p.add_argument("name"); p.add_argument("--tail", type=int, default=30); p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("collect"); p.add_argument("names", nargs="*"); p.add_argument("--hf-repo", default=None)
    p.add_argument("--runs", default="runs"); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("destroy"); p.add_argument("names", nargs="*"); p.add_argument("--all", action="store_true")
    p.add_argument("--yes", action="store_true"); p.set_defaults(fn=cmd_destroy)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
