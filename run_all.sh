#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p paper/generated
case "${1:-quick}" in
  quick) python theory/verify_nq.py ;;
  tables)
    for script in make_r7_final make_r6_cells make_r7_n1_table make_r7_cost_table make_r8_table make_r9_table make_r10_table r7_reader_channel; do
      python "data/$script.py"
    done
    for ref in reference/*.tex; do cmp "$ref" "paper/generated/$(basename "$ref")"; done ;;
  figures) for n in 2 3 4; do python "figsrc/plot_fig$n.py"; done ;;
  *) echo 'Usage: bash run_all.sh {quick|tables|figures}' >&2; exit 2 ;;
esac
