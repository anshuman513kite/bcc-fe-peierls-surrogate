"""Read RESULT lines from the elastic logs and print C11, C12, C44, B, C', A."""
import re, sys
REF = {"C11": 243.4, "C12": 145.0, "C44": 116.0}  # as given by Ansh, cited to Mendelev et al. 2003, Phil. Mag. 83, 3977 (potential 2)
pat = re.compile(r"^RESULT e=(\S+) C11=(\S+) C12=(\S+) C44=(\S+) GPa", re.M)
print(f"{'e':>6} {'C11':>9} {'C12':>9} {'C44':>9} {'B':>9} {'Cprime':>9} {'A':>7}")
rows = {}
for f in sys.argv[1:]:
    e, c11, c12, c44 = map(float, pat.search(open(f).read()).groups())
    B, Cp, A = (c11 + 2*c12)/3, (c11 - c12)/2, 2*c44/(c11 - c12)
    rows[e] = dict(C11=c11, C12=c12, C44=c44)
    print(f"{e:6.3f} {c11:9.4f} {c12:9.4f} {c44:9.4f} {B:9.4f} {Cp:9.4f} {A:7.4f}")
for k, ref in REF.items():
    vals = [r[k] for r in rows.values()]
    print(f"{k}: spread between strains {max(vals)-min(vals):.4f} GPa; "
          + "; ".join(f"e={e}: diff {r[k]-ref:+.4f} GPa ({100*(r[k]-ref)/ref:+.3f}%)" for e, r in rows.items()))
