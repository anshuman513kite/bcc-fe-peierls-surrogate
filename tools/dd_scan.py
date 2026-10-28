"""DD scan of a Peierls ramp (Session 11): Burgers-triangle site of every saved step, and the first stress at which it left
the starting site.

Why: the ramp driver (tools/peierls_chi.py) decides "moved" from the excess-energy tracker (tools/track_core.py) with the
threshold |shift| >= one jump (2.3313 A). At chi = -30 / -24 a jump of the core by one site onto the +60 deg {110} plane
showed up in the tracker as only ~1.98 A (the energy centroid is not the DD centroid), so it was not flagged. The DD
Burgers triangle (tools/ddmap_data.py) is the direct test of Ansh's definition (first stress at which no stable state
remains in the starting valley): a step "left" if its triangle centroid differs from the start by > 0.01 A.

Usage (project root): python tools/dd_scan.py <run folder> [<run folder> ...] [--nz 21] [--jobs 6]
Writes <run folder>/dd_scan.csv (kind, tau, centroid dy dz, orientation, top column sum) and prints a summary line per folder:
DDSCAN <name> start=(dy,dz) first_left=<tau or none> (<kind>) shift=(dy,dz) angle=<deg from +z> last_at_start=<tau>
"""
import argparse, csv, glob, os, re, subprocess, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("folders", nargs="+"); ap.add_argument("--nz", default="21"); ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--png_dir", default=os.environ.get("TMPDIR", "/tmp"), help="scratch folder for the (discarded) DD map images")
a = ap.parse_args()


def dd(args):
    folder, path = args
    out = subprocess.run([sys.executable, f"{ROOT}/tools/ddmap_data.py", path, os.path.join(a.png_dir, f"dd_{os.getpid()}.png")],
                         capture_output=True, text=True, cwd=folder, env=dict(os.environ, SCREW_NZ=a.nz, MPLBACKEND="Agg"))
    m = re.search(r"dy = (\S+) A, dz = (\S+) A .*orientation '(\w+)'", out.stdout)
    c = re.search(r"column \(dy, dz\) = .*? sum \|DD\| on its bonds = (\S+) b", out.stdout)
    if not m:
        return path, None
    return path, (float(m.group(1)), float(m.group(2)), m.group(3), float(c.group(1)))


for folder in a.folders:
    folder = os.path.abspath(folder)
    files = [f for f in glob.glob(f"{folder}/steps/*.data") if not f.endswith(".in.data")]
    key = lambda f: (float(re.search(r"_(\d+\.\d+)\.data$", f).group(1)), 0 if "coarse" in f else 1)
    files.sort(key=key)
    with Pool(a.jobs) as p:
        res = dict(p.map(dd, [(folder, os.path.relpath(f, folder)) for f in files]))
    rows = []
    for f in files:
        r = res[os.path.relpath(f, folder)]
        if r is None:
            print(f"DD failed for {f}"); continue
        rows.append(dict(kind=os.path.basename(f).split("_")[0], tau_GPa=key(f)[0], dd_dy_A=r[0], dd_dz_A=r[1], orientation=r[2], topcol_b=r[3]))
    with open(f"{folder}/dd_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    s = rows[0]
    left = lambda r: np.hypot(r["dd_dy_A"] - s["dd_dy_A"], r["dd_dz_A"] - s["dd_dz_A"]) > 0.01
    byt = sorted(rows, key=lambda r: r["tau_GPa"])
    first = next((r for r in byt if left(r)), None)
    at = [r["tau_GPa"] for r in byt if not left(r) and (first is None or r["tau_GPa"] < first["tau_GPa"])]
    if first:
        dy, dz = first["dd_dy_A"] - s["dd_dy_A"], first["dd_dz_A"] - s["dd_dz_A"]
        print(f"DDSCAN {os.path.basename(folder)} start=({s['dd_dy_A']:+.3f},{s['dd_dz_A']:+.3f}) first_left={first['tau_GPa']:.3f} ({first['kind']}) "
              f"shift=({dy:+.3f},{dz:+.3f}) |shift|={np.hypot(dy, dz):.3f} angle={np.degrees(np.arctan2(dy, dz)):+.0f} last_at_start={max(at):.3f} "
              f"topcol={first['topcol_b']:.3f}b n={len(rows)}")
    else:
        print(f"DDSCAN {os.path.basename(folder)} never left; n={len(rows)}")
