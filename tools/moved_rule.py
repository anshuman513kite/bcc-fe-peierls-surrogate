""""Moved" rule for the Peierls ramps: Ansh's D5 (28 Oct, Session 14), replacing D1 (Session 13).

D5: the compact easy core at the start site is lost ("moved") at the first step where EITHER
  (a) glide jump:   the DD Burgers-triangle site moved by >= one jump (JUMP - 0.01 A) to a 'down' site whose top column
                    is < 1.4 b (a compact easy core at a new site), OR
  (b) easy -> split: the core becomes column-centred (top column >= 1.4 b), wherever the DD finder puts the triangle.
Reason (Ansh): the definition is "no stable state left in the starting valley". The split core sits on a vertex column of
the starting triangle (1.35 A from the start), not in the starting easy-core minimum. Rule (b) does not depend on which
neighbouring triangle the finder names: D1 used that label and fired on a pure relabelling at chi = +28 (2.170 GPa,
<= 0.013 A of atomic motion).
A site change without either condition (a compact core shifted by less than one jump) is an 'anomaly' (logged, not moved).
same_state(): label-free comparison for the stability test: both compact at the same site, or both column-centred on the
same column.
History: D1 (Session 13) = classify_d1; Session 11-12 rule = any site change > 0.01 A.
"""
import os, re, subprocess, sys
import numpy as np

JUMP = 2.8553 * np.sqrt(6) / 3        # 2.3313 A, a*sqrt(6)/3
TOL = 0.01                            # A, site tolerance
SPLIT_B = 1.4                         # b, top-column |DD| sum: compact easy core 1.10-1.23 b, split states 1.54-1.80 b
COL_TOL = 0.1                         # A, same column


def classify(dy, dz, orientation, topcol):
    """D5 event of a relaxed state, relative to the compact start site: 'none', 'glide_jump', 'split' or 'anomaly'."""
    if topcol >= SPLIT_B:
        return "split"
    shift = float(np.hypot(dy, dz))
    if shift <= TOL:
        return "none"
    if shift >= JUMP - TOL and orientation == "down":
        return "glide_jump"
    return "anomaly"


def is_moved(event):
    return event in ("glide_jump", "split")


EVENT_NAME = {"glide_jump": "glide jump", "split": "easy -> split transition", "none": "", "anomaly": "anomaly"}


def same_state(a, b):
    """a, b: dicts from dd_state(). Same resting state, independent of the triangle label."""
    sa, sb = a["topcol"] >= SPLIT_B, b["topcol"] >= SPLIT_B
    if sa != sb:
        return False
    if sa:
        return float(np.hypot(a["col"][0] - b["col"][0], a["col"][1] - b["col"][1])) < COL_TOL
    return float(np.hypot(a["dy"] - b["dy"], a["dz"] - b["dz"])) < TOL


def classify_d1(dy, dz, orientation="down"):
    """D1 (Session 13), kept to reproduce Session 13 runs: moved / core_change / none from the triangle site only."""
    shift = float(np.hypot(dy, dz))
    if shift <= TOL:
        return "none"
    return "moved" if shift >= JUMP - TOL and orientation == "down" else "core_change"


def dd_state(path, cwd=".", nz=21, png=None):
    """Run tools/ddmap_data.py on a data file: triangle centroid (dy, dz) rel. to the inserted core, orientation, top column."""
    root = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, f"{root}/ddmap_data.py", path, png or os.path.join(cwd, "steps", "_dd_site.png")],
                         capture_output=True, text=True, cwd=cwd, env=dict(os.environ, SCREW_NZ=str(nz), MPLBACKEND="Agg"), check=True).stdout
    m = re.search(r"dy = (\S+) A, dz = (\S+) A .*orientation '(\w+)'", out)
    c = re.search(r"column \(dy, dz\) = \((\S+), (\S+)\) A: sum \|DD\| on its bonds = (\S+) b", out)
    return dict(dy=float(m.group(1)), dz=float(m.group(2)), orient=m.group(3), topcol=float(c.group(3)),
                col=(float(c.group(1).rstrip(",")), float(c.group(2))))
