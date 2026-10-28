"""Physics regression tests: read the existing, committed outputs (no LAMMPS run) and re-check the key numbers.
References: Mendelev et al., Phil. Mag. 83, 3977 (2003), potential 2 (a0 2.855 A; C11/C12/C44 243.4/145.0/116.0 GPa as given by Ansh)."""
import os, re, subprocess, sys
import numpy as np
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JUMP = 2.8553 * np.sqrt(6) / 3


def test_lattice_constant():
    a = float(re.search(r"^RESULT a0 = (\S+) Angstrom", open(f"{ROOT}/checks/01_lattice/log.lattice").read(), re.M).group(1))
    assert abs(a - 2.855) / 2.855 < 0.001          # computed 2.85532 A (+0.011 %)


@pytest.mark.parametrize("log", ["log.elastic_e0.001", "log.elastic_e0.005"])
def test_elastic_constants(log):
    m = re.search(r"^RESULT e=(\S+) C11=(\S+) C12=(\S+) C44=(\S+) GPa", open(f"{ROOT}/checks/02_elastic/{log}").read(), re.M)
    c11, c12, c44 = map(float, m.groups()[1:])
    for c, ref in ((c11, 243.4), (c12, 145.0), (c44, 116.0)):
        assert abs(c - ref) / ref < 0.01


def _build(nz):
    code = "import build, numpy as np; print(build.B, len(build.perfect_lattice()))"
    out = subprocess.run([sys.executable, "-c", code], cwd=f"{ROOT}/sims/01_screw_cell", env=dict(os.environ, SCREW_NZ=str(nz)),
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), int(out[1])


def test_burgers_vector():
    b, _ = _build(21)
    assert abs(b - 2.4728) < 1e-4


@pytest.mark.parametrize("nz, n", [(21, 27972), (42, 55944)])
def test_screw_cell_atom_count(nz, n):
    assert _build(nz)[1] == n


def _track(pe, yp, zp):
    LZ = 42 * 2.8553 * np.sqrt(6)
    out = subprocess.run([sys.executable, f"{ROOT}/tools/track_core.py", pe, str(yp), str(zp), str(LZ)],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"y=(\S+) z=(\S+)", out)
    return float(m.group(1)), float(m.group(2))


def test_tracker_detects_known_jump():
    """2x cell, chi = 0: 1.270 GPa is before the move, 1.280 GPa after (Session 7, DD map). Per-atom energies from
    tools/run_track.sh (git-ignored); skipped if they are not on disk."""
    d = f"{ROOT}/sims/03_peierls_chi0_2x/track"
    if not os.path.exists(f"{d}/fine_1.280.pe"):
        pytest.skip("per-atom energy dumps not present (run tools/run_track.sh)")
    y0, z0 = np.load(f"{ROOT}/sims/03_peierls_chi0_2x/screw_down.data.core.npy")
    y1, z1 = _track(f"{d}/fine_1.270.pe", y0, z0)
    y2, z2 = _track(f"{d}/fine_1.280.pe", y1, z1)
    assert np.hypot(y1 - y0, z1 - z0) < JUMP          # no jump yet at 1.270 GPa
    assert np.hypot(y2 - y0, z2 - z0) >= JUMP         # left the starting valley at 1.280 GPa
