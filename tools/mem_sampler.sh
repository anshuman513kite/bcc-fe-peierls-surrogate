#!/bin/bash
# Memory sampler (v0.2 step 2, 2 Nov 2026), started by tools/launch_safe.sh next to every run.
# Usage: mem_sampler.sh <watched_pid> <log_file> [process_name (default lmp)] [interval_s (default 10)]
# While <watched_pid> is alive, every interval it appends one line to <log_file>:
#   HH:MM:SS MEM <name>: processes=N rss_sum=X MB peak=Y MB available=Z MB
# rss_sum = sum of the resident memory (RSS) of all processes whose command name is <name> (ps -C), i.e. all MPI ranks.
# Shared pages count once per process, so the sum is an upper bound of the real use. available = MemAvailable of the machine.
# When the run ends it writes a final "MEM peak" line.
pid=$1; log=$2; name=${3:-lmp}; dt=${4:-10}
peak=0
while kill -0 "$pid" 2>/dev/null; do
    read n kb < <(ps -C "$name" -o rss= | awk '{s += $1; n++} END {print n + 0, s + 0}')
    mb=$(( kb / 1024 )); (( mb > peak )) && peak=$mb
    avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    echo "$(date +%H:%M:%S) MEM $name: processes=$n rss_sum=$mb MB peak=$peak MB available=$avail MB" >> "$log"
    sleep "$dt"
done
echo "$(date +%H:%M:%S) MEM peak ($name, sum of RSS over all processes, sampled every $dt s): $peak MB" >> "$log"
