"""Sequential job queue with resume and per-point status (v0.2 step 2, 2 Nov 2026), for the overnight grids (T, then P).

Usage (launch with tools/launch_safe.sh, so it is detached and crash-safe):
  python grid_queue.py <jobs.json> <status.json> <run_name>
jobs.json: a list of {"id": "...", "cwd": "...", "cmd": ["...", ...]}; jobs run one at a time, in order (CLAUDE.md: one
LAMMPS run at a time). status.json maps id -> {"state": pending|running|done|failed, "rc", "start", "end", "seconds"}; it is
rewritten atomically (temporary file + rename) after every change, so a kill can never leave it half-written.
Resume: start the same command again. Jobs in state "done" are skipped; a job left "running" by a killed queue is run again
from the start (its partial output is overwritten by the job itself); "failed" jobs are not retried unless --retry_failed.
Progress (jobs done / total, elapsed, estimated time left from the mean time of the jobs done in this session) goes to
logs/<run_name>.log. Exit status 0 if every job is done, 1 otherwise.
"""
import argparse, json, os, subprocess, sys, time

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_status(path, jobs):
    st = json.load(open(path)) if os.path.exists(path) else {}
    for j in jobs:
        st.setdefault(j["id"], {"state": "pending"})
    return st


def save_status(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"); ap.add_argument("status"); ap.add_argument("name")
    ap.add_argument("--retry_failed", action="store_true")
    ap.add_argument("--logdir", default=os.path.join(root, "logs"))
    a = ap.parse_args(argv)
    jobs = json.load(open(a.jobs))
    ids = [j["id"] for j in jobs]
    assert len(set(ids)) == len(ids), "job ids must be unique"
    st = load_status(a.status, jobs)
    os.makedirs(a.logdir, exist_ok=True)
    plog = open(os.path.join(a.logdir, f"{a.name}.log"), "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); plog.write(line + "\n"); plog.flush()

    resumed = [i for i in ids if st[i]["state"] == "running"]
    for i in resumed:
        st[i]["state"] = "pending"
    if resumed:
        say(f"resume: {len(resumed)} job(s) left 'running' by a stopped queue are run again: {', '.join(resumed)}")
    if a.retry_failed:
        for i in ids:
            if st[i]["state"] == "failed":
                st[i]["state"] = "pending"
    save_status(a.status, st)
    t0, times = time.time(), []
    todo = [j for j in jobs if st[j["id"]]["state"] == "pending"]
    say(f"queue {a.name}: {len(jobs)} jobs, {sum(st[i]['state'] == 'done' for i in ids)} done, {len(todo)} to run")
    for j in todo:
        i = j["id"]
        st[i].update(state="running", start=time.strftime("%Y-%m-%d %H:%M:%S")); save_status(a.status, st)
        ts = time.time()
        rc = subprocess.run(j["cmd"], cwd=j.get("cwd") or None).returncode
        dt = time.time() - ts; times.append(dt)
        st[i].update(state="done" if rc == 0 else "failed", rc=rc, end=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(dt, 1))
        save_status(a.status, st)
        n_done = sum(st[k]["state"] == "done" for k in ids)
        left = sum(st[k]["state"] == "pending" for k in ids)
        say(f"job {i}: {st[i]['state']} (rc {rc}, {dt:.0f} s) | done {n_done} / {len(ids)} | elapsed {time.time() - t0:.0f} s | "
            f"estimated left {left * sum(times) / len(times) / 60:.0f} min")
    n_done = sum(st[k]["state"] == "done" for k in ids)
    say(f"queue {a.name} finished: {n_done} / {len(ids)} done, {sum(st[k]['state'] == 'failed' for k in ids)} failed")
    return 0 if n_done == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
