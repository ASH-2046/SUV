Vendored NAVSIM v2 devkit files used by SUV NAVSIM v2 evaluation.

These files are derived from the official
[NAVSIM repository](https://github.com/autonomousvision/navsim) and remain
licensed under the Apache License 2.0. See `LICENSE` in this directory.

This directory contains an official NAVSIM devkit snapshot so
`experiments/navsimv2/scripts/evaluation/run_epdm_score.sh`
can call the official scorer entrypoints directly:

- `navsim/planning/script/run_pdm_score.py` for `navhard_two_stage`
- `navsim/planning/script/run_pdm_score_one_stage.py` for `navtest`
