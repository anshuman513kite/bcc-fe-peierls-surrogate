"""Argument parser of tools/stability_test.py, in its own module (3 Nov 2026) so a driver can check an exact argv without running
the test (run_step4.py dry run). Definitions unchanged from stability_test.py. A value starting with '-' that is not a plain
number (e.g. --model_key -30,+0.75) must be passed as --opt=value, or argparse reads it as a new option."""
import argparse, os


def make_parser():
    ap = argparse.ArgumentParser(prog="stability_test.py")
    ap.add_argument("data"); ap.add_argument("tau", type=float); ap.add_argument("chi", type=float); ap.add_argument("y0", type=float)
    ap.add_argument("tag")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--np", type=int, default=int(os.environ.get("NP", 7)))
    ap.add_argument("--nz", type=int, default=21); ap.add_argument("--model", default=None)
    ap.add_argument("--ref", default=None, help="unstrained reference data file for the box (default: the state itself, chi = 0)")
    ap.add_argument("--jiggle", default="0.001"); ap.add_argument("--etol", default="0"); ap.add_argument("--ftol", default="1e-6")
    ap.add_argument("--syz", type=float, default=0.0); ap.add_argument("--model_key", default=None)
    ap.add_argument("--dmax", default="0.01"); ap.add_argument("--maxit", type=int, default=200000); ap.add_argument("--maxeval", type=int, default=2000000)
    return ap
