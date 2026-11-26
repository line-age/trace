"""Synthetic checks for score.py: known inputs, known answers. ./qpy debug/check_score.py"""

import sys
from pathlib import Path

import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import score  # noqa: E402

L = shapely.LineString
area = shapely.box(-1000, -1000, 21000, 11000)
buf = 50.0
truth = [L([(0, 0), (20000, 0)])]
fails = 0


def check(name, got, want, tol=0.01):
    global fails
    ok = got == want if not isinstance(want, float) else abs(got - want) <= tol
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got}, want {want}")


m, _ = score(truth, truth, area, buf)
check("identical: recall", m["length_recall"], 1.0)
check("identical: precision", m["length_precision"], 1.0)
check("identical: breaks", m["breaks"], 0)

m, _ = score([L([(0, 200), (20000, 200)])], truth, area, buf)
check("shifted 4 buffers: recall", m["length_recall"], 0.0)
check("shifted 4 buffers: precision", m["length_precision"], 0.0)

m, _ = score([L([(0, 0), (9000, 0)]), L([(11000, 0), (20000, 0)])], truth, area, buf)
check("split with 2 km gap: recall", m["length_recall"], 0.9)
check("split: breaks", m["breaks"], 2)
check("split: missing km (gap minus a buffer at each end)", m["missing_length_km"], 1.9, tol=0.025)  # sampled every buf/4

m, _ = score([L([(0, 0), (10000, 0)])], truth, area, buf)
check("half found: recall", m["length_recall"], 0.5)
check("half found: precision", m["length_precision"], 1.0)
check("half found: one break", m["breaks"], 1)

half = [L([(0, 0), (10000, 0)])]
far = [shapely.Point(50000, 50000)]  # hand-marked, but in another area
m, _ = score(half, truth, area, buf, true_ends=[])
check("no hand-marked ends (only junctions marked): falls back to truth ends", m["breaks"], 1)
m, _ = score(half, truth, area, buf, true_ends=far)
check("hand-marked ends only in another area: falls back", m["breaks"], 1)
m, _ = score(half, truth, area, buf, true_ends=[shapely.Point(0, 0), shapely.Point(10000, 0)])
check("hand-marked ends where the traced line stops: no break", m["breaks"], 0)

two = [L([(0, 0), (8000, 0)]), L([(12000, 5000), (20000, 5000)])]
m, _ = score([L([(0, 0), (8000, 0), (12000, 5000), (20000, 5000)])], two, area, buf)
check("joins two separate lines: false joins", m["false_joins"], 1)

m, _ = score([L([(0, 0), (10000, 0), (10000, 8000)])], [L([(0, 0), (10000, 0)])], area, buf)
check("runs on into non-target: false joins", m["false_joins"], 1)

pair = [L([(0, 0), (20000, 0)]), L([(0, 40), (20000, 40)])]  # parallels closer than the buffer
m, _ = score([pair[0]], pair, area, buf)
check("parallel pair, one traced: recall", m["length_recall"], 0.5)
m, _ = score(pair, pair, area, buf)
check("parallel pair, both traced: recall", m["length_recall"], 1.0)

tee = [L([(0, 0), (20000, 0)]), L([(10000, 0), (10000, 8000)])]
m, _ = score(tee, tee, area, buf)
check("T junction found", m["junction_recall"], 1.0)
m, _ = score([L([(0, 0), (20000, 0)])], tee, area, buf)
check("T junction missing branch", m["junction_recall"], 0.0)
m, _ = score(tee, tee, area, buf, true_junctions=[])
check("no hand-marked junctions (only ends marked): falls back", m["junctions"], 1)
m, _ = score([L([(0, 0), (10000, 0), (10000, 8000)]), L([(10000, 0), (20000, 0)])], tee, area, buf)
check("traced line turning at a real T junction: not a false join", m["false_joins"], 0)

m, _ = score([L([(-1000, 0), (21000, 0)])], [L([(-1000, 0), (21000, 0)])], area, buf)
check("line crossing the area edge: no breaks", m["breaks"], 0)

pool = shapely.box(8000, -3000, 12000, 3000)  # a pool cut out of the area (a hole)
holed = area.difference(pool)
m, _ = score([L([(0, 0), (8000, 0)]), L([(12000, 0), (20000, 0)])], truth, holed, buf)
check("pool: stops at both pool edges: recall", m["length_recall"], 1.0)
check("pool: stops at both pool edges: breaks", m["breaks"], 0)
m, _ = score(truth, [L([(0, 0), (8000, 0)]), L([(12000, 0), (20000, 0)])], holed, buf)
check("pool: crosses a pool the truth stops at: precision", m["length_precision"], 1.0)
m, _ = score([L([(0, 0), (5000, 0)])], truth, holed, buf)
check("pool: a real break outside the pool still counts", m["breaks"], 1)
check("pool: missing outside the pool still counts (km)", m["missing_length_km"], 10.95, tol=0.025)  # less a buffer at the break

print("all passed" if not fails else f"{fails} failed")
sys.exit(bool(fails))
