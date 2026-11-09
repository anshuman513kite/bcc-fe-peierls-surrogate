"""Render a LAMMPS input the way LAMMPS substitutes ${name}, and check its numeric parameters (v0.2, 3 Nov 2026; pure Python).

render(path, vars): 'variable X index V' sets X unless given on the command line (vars, i.e. -var); ${name} is replaced;
comments and blank lines are dropped; $(...) immediate expressions are left as written. Returns the command lines.
cmd_vars(cmd): the -var name value pairs of an mpirun/lmp command list.
bad_numbers(lines, cmd): numeric parameters that would not parse as a float in LAMMPS: the force components of every
fix ... addforce line and every numeric -var value; rejects '--', '+-', '-+', nan, inf. Returns a list of problems.
"""
import math, re

NON_NUMERIC_VARS = {"pot", "datain", "dataout", "peout", "groups"}


def render(path, over):
    vals, out = dict(over), []
    for raw in open(path):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"variable\s+(\S+)\s+index\s+(\S+)", line)
        if m:
            vals.setdefault(m.group(1), re.sub(r"\$\{(\w+)\}", lambda mm: str(vals.get(mm.group(1), mm.group(0))), m.group(2)))
        out.append(re.sub(r"\$\{(\w+)\}", lambda mm: str(vals.get(mm.group(1), mm.group(0))), line))
    return out


def cmd_vars(cmd):
    return {cmd[i + 1]: cmd[i + 2] for i, a in enumerate(cmd) if a == "-var"}


def _is_float(tok):
    if re.search(r"--|\+-|-\+|\+\+", tok) or tok.lower().lstrip("+-") in ("nan", "inf", "infinity"):
        return False
    try:
        return math.isfinite(float(tok))
    except ValueError:
        return False


def bad_numbers(lines, cmd=None):
    bad = []
    for l in lines:
        w = l.split()
        if len(w) >= 7 and w[0] == "fix" and w[3] == "addforce":
            bad += [f"{l}: '{t}'" for t in w[4:7] if not _is_float(t)]
    for k, v in (cmd_vars(cmd).items() if cmd else ()):
        if k not in NON_NUMERIC_VARS and not _is_float(v):
            bad.append(f"-var {k} '{v}'")
    return bad
