#!/bin/bash
# Download the interatomic potential used by Project A and check its SHA-256 (the potential file is not in the repository).
#
# Potential: Mendelev, Han, Srolovitz, Ackland, Sun & Asta, Phil. Mag. 83, 3977-3994 (2003), potential #2 (Fe), eam/fs format.
# Default source: NIST Interatomic Potentials Repository, entry 2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2, version 1
#   (Fe_2.eam.fs, dated 10 Jun 2007). Its 30,000 tabulated values are identical to the file used for every run here; only
#   the three comment lines at the top differ (checked with tools/compare_eamfs.py, NOTES.md Session 4). Version 3 (2009) of the
#   same NIST entry has different tables and is NOT the one used.
# --lammps: the byte-identical file used for the runs (Fe_mm.eam.fs from the LAMMPS repository, stable branch).
# Either way the file is installed as potentials/Fe_mm.eam.fs, the name the inputs use.
# Usage (from the project root): tools/fetch_potential.sh [--lammps]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p potentials
if [ "${1:-}" = "--lammps" ]; then
  url=https://raw.githubusercontent.com/lammps/lammps/stable/potentials/Fe_mm.eam.fs
  sha=d7b15f435a3f5192e8ed264d0dcbc052f1d5641ba5db610eab88199d77d20f7c
else
  url=https://www.ctcms.nist.gov/potentials/Download/2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2/1/Fe_2.eam.fs
  sha=e3c12de551e59e1b94872201c0fc1889a40ea2f90ff49d205842e9158f245eda
fi
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp"
got=$(sha256sum "$tmp" | cut -d' ' -f1)
if [ "$got" != "$sha" ]; then
  echo "SHA-256 mismatch for $url"; echo "  expected $sha"; echo "  got      $got"; exit 1
fi
mv "$tmp" potentials/Fe_mm.eam.fs
trap - EXIT
echo "OK: potentials/Fe_mm.eam.fs from $url (SHA-256 $sha)"
