#!/usr/bin/env bash
# Full experiment: train the three required modes (+ stretch modes) and evaluate.
# Usage: bash scripts/run_all.sh            (core: wlt_only, clt_only, mtt)
#        STRETCH=1 bash scripts/run_all.sh  (also selfesc, learned router, rule_words control)
# Extra flags are passed to every training run, e.g.:  EXTRA="--steps 3000" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA="${EXTRA:-}"

python -m pytest -q
python -m mtt.train --mode wlt_only $EXTRA
python -m mtt.train --mode clt_only $EXTRA
python -m mtt.train --mode mtt $EXTRA
RUNS="wlt_only clt_only mtt"
if [ "${STRETCH:-0}" = "1" ]; then
  python -m mtt.train --mode selfesc $EXTRA
  python -m mtt.train --mode mtt --router learned $EXTRA
  python -m mtt.train --mode mtt --router rule_words --name mtt_rule_words $EXTRA
  RUNS="$RUNS mtt_rule_words mtt_learned selfesc"
  python -m mtt.eval --router-compare
fi
python -m mtt.eval --runs $RUNS
