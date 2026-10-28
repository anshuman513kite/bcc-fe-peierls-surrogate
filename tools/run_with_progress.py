"""Run a LAMMPS minimisation and write progress to logs/<run_name>.log every few seconds.

Usage: python run_with_progress.py <run_name> <max_iterations> -- <lammps command...>
The LAMMPS command must write its log with -log <file>; that file is parsed for thermo lines.
Progress line: iteration done / max, elapsed time, estimated time left (upper bound, assumes the max is reached).
"""
import subprocess, sys, time, os, re

name, maxit = sys.argv[1], int(sys.argv[2])
cmd = sys.argv[sys.argv.index("--") + 1:]
lmp_log = cmd[cmd.index("-log") + 1]
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")   # <project>/logs, from any cwd
os.makedirs(log_dir, exist_ok=True)
plog = os.path.join(log_dir, f"{name}.log")
t0 = time.time()
p = subprocess.Popen(cmd)
step_re = re.compile(r"^\s+(\d+)\s+-?\d")
with open(plog, "w") as out:
    out.write(f"run {name} started {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
    while p.poll() is None:
        time.sleep(5)
        step = 0
        try:
            in_min = False
            for line in open(lmp_log):
                if line.startswith("minimize"):
                    in_min = True
                m = step_re.match(line)
                if in_min and m:
                    step = int(m.group(1))
        except FileNotFoundError:
            pass
        el = time.time() - t0
        left = el / step * (maxit - step) if step else float("nan")
        out.write(f"{time.strftime('%H:%M:%S')} iteration {step} / {maxit} (max) | elapsed {el:6.0f} s | "
                  f"time left <= {left:6.0f} s (upper bound)\n")
        out.flush()
    out.write(f"finished, exit code {p.returncode}, total {time.time() - t0:.0f} s\n")
sys.exit(p.returncode)
