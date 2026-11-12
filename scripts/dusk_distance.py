"""Can a learner get home in time? Two things, both from the visualizer's data (run viz/build.py first):

1. The maps: steps (forward or turn, 1 each) between the shelter, water and food in each of the 32 recorded worlds,
   against the 26 steps left when the adult first says "home" (12 dusk steps + 14 night steps survivable at drain 1/15).
2. The recordings: for every night that started outside the shelter, how far the learner was from the shelter when
   dusk started, and whether it reached the shelter and whether it died of cold that night.

usage: python scripts/dusk_distance.py [run@snapshot,snapshot ...]     default: the shelter grid's later snapshots"""
import base64
import collections
import json
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = "viz/data"
ix = json.load(open(f"{D}/index.json")); W = json.load(open(f"{D}/worlds.json")); G = W["grid"]
FI = {f: i for i, f in enumerate(ix["fields"])}
DXY = [(0, -1), (1, 0), (0, 1), (-1, 0)]
EMPTY, FOOD, WATER, TREE, ROCK, SHELTER, PRED, HAZ = range(8)
ter = [np.frombuffer(base64.b64decode(w["terrain"]), np.uint8).reshape(G, G) for w in W["worlds"]]

def dist_to(t, goals):
    """steps (forward or turn = 1 each) from every (x, y, facing) to the nearest goal cell; hazards, trees, rocks avoided."""
    INF = 10**6; d = np.full((G, G, 4), INF); q = collections.deque()
    for (gx, gy) in goals:
        for f in range(4): d[gy, gx, f] = 0; q.append((gx, gy, f))
    while q:                                   # reverse BFS: predecessors of (x, y, f)
        x, y, f = q.popleft(); c = d[y, x, f] + 1
        for pf in ((f + 1) % 4, (f - 1) % 4):  # arrived by turning
            if d[y, x, pf] > c: d[y, x, pf] = c; q.append((x, y, pf))
        px, py = x - DXY[f][0], y - DXY[f][1]  # arrived by moving forward
        if 0 <= px < G and 0 <= py < G and t[py, px] not in (TREE, ROCK, HAZ) and d[py, px, f] > c:
            d[py, px, f] = c; q.append((px, py, f))
    return d

BUDGET = 12 + 14          # dusk steps + night steps a learner at full warmth survives unsheltered at drain 1/15
rows = []
for w in range(W["n"]):
    t = ter[w]
    cells = lambda k: [(int(x), int(y)) for y, x in zip(*np.where(t == k))]
    sh = cells(SHELTER); dS = dist_to(t, sh).min(2)       # best facing
    wat, food = cells(WATER), cells(FOOD)
    dw = [int(dS[y, x]) for x, y in wat]; df = [int(dS[y, x]) for x, y in food]
    dWat = dist_to(t, wat).min(2); dFood = dist_to(t, food).min(2)
    wf = min(int(dWat[y, x]) for x, y in food)
    spawn_to_sh = int(dS[G // 2, G // 2])
    # shortest daily loop shelter -> water -> food -> shelter (either order), lower bound (best facings)
    loop = min(min(dw) + wf + min(df), min(df) + wf + min(dw)) if dw and df else None
    rows.append(dict(w=w, shelter=sh[0] if sh else None, water_to_shelter=min(dw), food_to_shelter=(min(df), max(df)), water_to_food=wf,
                     spawn_to_shelter=spawn_to_sh, loop=loop, far_cells=float((dS[t != TREE] > BUDGET)[(dS[t != TREE] < 10**6)].mean())))
print("world shelter | water->shelter  nearest/farthest food->shelter  water<->food  spawn->shelter  loop(shelter-water-food-shelter)")
for r in rows: print(r["w"], r["shelter"], "|", r["water_to_shelter"], r["food_to_shelter"], r["water_to_food"], r["spawn_to_shelter"], r["loop"])
a = lambda k: np.array([r[k] for r in rows])
print(f"\nbudget after the adult first says 'home' (dusk start): {BUDGET} steps")
print("worlds where water is farther than that from the shelter:", int((a("water_to_shelter") > BUDGET).sum()), "of", len(rows))
print("worlds where the NEAREST food is farther than that:", int((np.array([r['food_to_shelter'][0] for r in rows]) > BUDGET).sum()))
print("worlds where water OR nearest food is farther:", int(((a("water_to_shelter") > BUDGET) | (np.array([r['food_to_shelter'][0] for r in rows]) > BUDGET)).sum()))
print("loop length shelter->water->food->shelter: median", np.median(a("loop")), "max", a("loop").max(), "| worlds with loop > 74 (one out-of-shelter window):", int((a("loop") > 74).sum()))
print("share of walkable cells farther than the budget from the shelter: mean %.2f" % a("far_cells").mean())



def analyse(run, snaps=("u00600",)):
    bins = [(0, 8), (9, 16), (17, 26), (27, 99)]; out = {b: [0, 0, 0] for b in bins}   # nights, sheltered, died of cold that night
    prog = []; home_said = []
    for snap in snaps:
        for e in json.load(open(f"{D}/ep/{run}@{snap}.json"))["episodes"]:
            u = np.frombuffer(base64.b64decode(e["data"]), np.uint8).reshape(-1, len(ix["fields"])); w = e["world"]
            dS3 = dist_to(ter[w], [(int(x), int(y)) for y, x in zip(*np.where(ter[w] == SHELTER))])
            dist = lambda i: int(dS3[u[i, FI["ly"]], u[i, FI["lx"]], u[i, FI["lf"]]])
            for day in range(4):
                d0 = day * 80 + 48
                if len(u) <= d0 or u[d0, FI["sheltered"]]: continue
                d = dist(d0); b = next(b for b in bins if b[0] <= d <= b[1])
                night = u[day * 80 + 60: day * 80 + 80]
                sheltered = bool(len(night) and night[:, FI["sheltered"]].any())
                died_cold = e["cause"] == "cold" and len(u) <= day * 80 + 80
                out[b][0] += 1; out[b][1] += sheltered; out[b][2] += died_cold
                end = min(len(u) - 1, d0 + 26)
                first_sh = next((i for i in range(d0, end + 1) if u[i, FI["sheltered"]]), None)
                stop = first_sh if first_sh is not None else end
                if stop > d0: prog.append((d - dist(stop)) / (stop - d0))
                home_said.append(float((u[d0:stop + 1, FI["heard"]] == 15).mean()))
    print(f"{run} ({', '.join(snaps)}): nights started outside the shelter, by steps from shelter when dusk starts")
    for b in bins:
        n, s, c = out[b]
        if n: print(f"   {b[0]:2d}-{b[1]:2d} steps: {n:3d} nights | reached the shelter {100 * s / n:3.0f}% | died of cold that night {100 * c / n:3.0f}%")
    print(f"   progress toward the shelter after dusk starts: median {np.median(prog):.2f} steps closer per step (1.0 = straight there); heard 'home' on {100 * np.mean(home_said):.0f}% of those steps")

print()
for spec in sys.argv[1:] or ["sh_def_d15@u00400,u00600", "sh_rec_d15@u00400,u00600", "sh_def_d22@u00600", "sh_def_d35@u00600"]:
    run, snaps = spec.split("@")
    analyse(run, tuple(snaps.split(",")))
