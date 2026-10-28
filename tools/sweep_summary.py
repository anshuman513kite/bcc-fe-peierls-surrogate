"""Summarise the chi sweep (Session 11): data/peierls_chi_sweep.csv and figures/peierls_vs_chi.png.

Usage (project root): python tools/sweep_summary.py <baseline_GPa> [--stab logs/chi_sweep.out]
Per angle (data/chi_<cell>_<p|m><|chi|>.csv + logs/chi_<...>.log + stability lines):
  bracket (last unmoved, first moved tau), midpoint, resolved shear on (-110) at the bracket tau*cos(chi), Schmid prediction
  baseline/cos(chi) and the ratio midpoint/Schmid;
  glide plane from the DD Burgers-triangle shift between the last unmoved and first moved states: direction angle
  theta = atan2(dy, dz): ~0 -> (-110); ~+60 -> (0 1 -1); ~-60 -> (-1 0 1); ~+-30 -> zig-zag (two {110}, average {112});
  core shape: largest per-column |DD| sum at the last unmoved / first moved state (compact easy core ~1.10-1.21 b;
  a single column well above, e.g. 1.58 b, = core centred on one column);
  stability test at the last unmoved state (2 seeds): returned yes/no;
  sanity flags: plane other than (-110); bracket < 0.8 or > 3 GPa; any non-converged step below the bracket.
"moved" in the csv files comes from the DD Burgers-triangle site (peierls_chi.py --moved dd; runs before that fix were checked
with tools/dd_scan.py, which gives the same bracket for every angle except chi = -30, -24, which were rerun).
--stab: one file of STABILITY lines (concatenate the right lines from the sweep, rerun and fix logs).
"""
import argparse, csv, glob, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("baseline", type=float); ap.add_argument("--stab", default="logs/chi_sweep.out")
a = ap.parse_args()
PLANES = [(0, "(-110)"), (60, "(0 1 -1)"), (-60, "(-1 0 1)"), (30, "zig-zag +30 (avg {112})"), (-30, "zig-zag -30 (avg {112})"),
          (180, "(-110), backwards"), (120, "(-1 0 1), backwards"), (-120, "(0 1 -1), backwards")]
stab = {}
if os.path.exists(a.stab):
    for l in open(a.stab):
        m = re.match(r"STABILITY (\S+)_last_seed(\d) .*fmax=(\S+) .*returned=(\w+)", l)
        if m:
            stab.setdefault(m.group(1), []).append((m.group(4), float(m.group(3))))
rows = []
for f in sorted(glob.glob("data/chi_*_[pm]*.csv")):
    name = os.path.basename(f)[:-4]
    if not re.fullmatch(r"chi_(small|x2)_[pm]\d+", name):      # skip archived runs (*_v1finebug, *_trackermoved, *_dumpbug)
        continue
    r = list(csv.DictReader(open(f)))
    unm = [x for x in r if x["moved"] == "no"]; mov = [x for x in r if x["moved"] == "yes"]
    if not unm or not mov:
        continue
    chi = float(r[0]["chi_deg"]); lu = float(unm[-1]["tau_GPa"]); fm = min(float(x["tau_GPa"]) for x in mov)
    mid = (lu + fm) / 2; schmid = a.baseline / np.cos(np.radians(chi))
    log = open(f"logs/{name}.log").read()
    cen = {}
    for lab in ("last_unmoved", "first_moved"):
        m = re.search(rf"DD {lab}: .*triangle centroid relative to inserted core: dy = (\S+) A, dz = (\S+) A", log)
        c = re.search(rf"DD {lab}: .*column \(dy, dz\) = .*? sum \|DD\| on its bonds = (\S+) b", log)
        cen[lab] = (float(m.group(1)), float(m.group(2)), float(c.group(1)))
    dy, dz = cen["first_moved"][0] - cen["last_unmoved"][0], cen["first_moved"][1] - cen["last_unmoved"][1]
    theta = np.degrees(np.arctan2(dy, dz))
    plane = min(PLANES, key=lambda p: abs((theta - p[0] + 180) % 360 - 180))[1] if np.hypot(dy, dz) > 0.5 else "no DD shift"
    nonconv = [x["tau_GPa"] for x in unm if x["converged"] != "yes"]
    st = stab.get(name, [])
    flags = []
    if plane != "(-110)": flags.append("plane " + plane)
    if not 0.8 <= mid <= 3.0: flags.append("value out of 0.8-3 GPa")
    if nonconv: flags.append("non-converged below bracket at " + ",".join(nonconv))
    if st and any(s[0] != "yes" for s in st): flags.append("last unmoved state not stable")
    rows.append(dict(name=name, cell="2x" if "_x2_" in name else "small", chi_deg=chi,
                     side="twinning" if chi < 0 else ("anti-twinning" if chi > 0 else "-"), tau_last_unmoved=lu, tau_first_moved=fm,
                     tau_mid=mid, tau_cos_chi_mid=mid * np.cos(np.radians(chi)), schmid=schmid, ratio_to_schmid=mid / schmid,
                     dd_shift_dy=dy, dd_shift_dz=dz, dd_shift_angle_deg=theta, glide_plane=plane,
                     core_topcol_last_unmoved_b=cen["last_unmoved"][2], core_topcol_first_moved_b=cen["first_moved"][2],
                     stability_last_unmoved=";".join(f"{s[0]} (fmax {s[1]:.1e})" for s in st), flags="; ".join(flags)))
rows.sort(key=lambda x: (x["cell"], x["chi_deg"]))
with open("data/peierls_chi_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for x in rows:
    print(f"{x['cell']:5s} chi {x['chi_deg']:+5.0f} ({x['side']:13s}): tau_P {x['tau_last_unmoved']:.3f}-{x['tau_first_moved']:.3f} GPa | "
          f"tau cos chi {x['tau_cos_chi_mid']:.3f} | Schmid {x['schmid']:.3f} ratio {x['ratio_to_schmid']:.3f} | plane {x['glide_plane']} "
          f"(shift {x['dd_shift_angle_deg']:+.0f} deg) | core top col {x['core_topcol_last_unmoved_b']:.2f}/{x['core_topcol_first_moved_b']:.2f} b | "
          f"stable: {x['stability_last_unmoved']} | flags: {x['flags'] or '-'}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
cg = np.linspace(-32, 32, 200)
ax.axvspan(-32, 0, color="#2a78d6", alpha=0.06); ax.axvspan(0, 32, color="#eb6834", alpha=0.06)
ax.text(-31, 0.02, "twinning side (χ < 0)", transform=ax.get_xaxis_transform(), fontsize=7, color="#2a78d6")
ax.text(31, 0.02, "anti-twinning side (χ > 0)", transform=ax.get_xaxis_transform(), fontsize=7, color="#eb6834", ha="right")
ax.plot(cg, a.baseline / np.cos(np.radians(cg)), color="#1a1a19", lw=1, ls="--", label=f"Schmid: {a.baseline:.4f}/cos χ GPa (χ = 0 baseline)")
for cell, mk, c in (("small", "o", "#2a78d6"), ("2x", "s", "#eb6834")):
    s = [x for x in rows if x["cell"] == cell]
    if not s:
        continue
    ch = np.array([x["chi_deg"] for x in s]); lo = np.array([x["tau_last_unmoved"] for x in s]); hi = np.array([x["tau_first_moved"] for x in s])
    ax.errorbar(ch + (0.8 if cell == "2x" else 0), (lo + hi) / 2, yerr=(hi - lo) / 2, fmt=mk, color=c, mec="white", ms=6, capsize=3,
                label=f"{'146.87 Å cell' if cell == 'small' else '293.75 Å cell (2×, drawn 0.8° right)'}: bracket (last unmoved – first moved)")
ax.plot([0], [a.baseline], marker="D", color="#1a1a19", ms=5, ls="none", label="χ = 0 baseline (Gate B')")
ax.set_xlabel("MRSS angle χ (deg) from the (−110) plane"); ax.set_ylabel("Peierls stress τ_P (GPa, applied τ)")
ax.set_title("Screw dislocation in Fe (Mendelev 2003 #2): Peierls stress vs χ, 0 K\n(option-2 loading; force-converged CG, dmax 0.01 Å)", fontsize=9)
ax.set_xlim(-32, 32); ax.set_ylim(1.05, 2.15)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8); ax.legend(fontsize=7, frameon=False, loc="upper center")
fig.tight_layout(); fig.savefig("figures/peierls_vs_chi.png"); print("saved figures/peierls_vs_chi.png")
