# Notes for Claude sessions in this repository

- Research context and current results: read `RESEARCH_STATUS.md` first, and update it when
  results or the setting change. Usage and commands: `docs/coop-setting.md`.
- The design reference is `docs/CONTEXT_PARTNER_PLAN_v3.md` (context-dependent partners,
  proactive collaboration, order timers, Gate 0). It builds on `docs/PERSONA_PARTNER_PLAN_v2.md`
  and replaces v2's persona design, gates and pilot; where they disagree, v3 holds. v2's dates
  are not binding; v3's "report, then stop" points (after Gate 0 and after Gate 3) are. Say where
  we deviate.
- The research code is `src/coop/` (tests in `tests/test_coop_*.py`). The original benchmark
  (`src/main.py`, `evaluation.py`) and the env defaults must keep their original behaviour.
- Environment: Python 3.11 with `requirements-coop.txt` (`numpy<2`); run with
  `PYTHONPATH=src:lib/overcooked_ai`. Tests: `.venv/bin/python -m unittest discover -s tests`.
- The user runs experiments on a lab server (repo at `~/work/collab-overcooked`) with `lab-run`.
  End every update with the commands to pull the branch and the `lab-run` commands to run:
  CPU jobs `lab-run --new --gpus 0 -- ...`; LLM jobs via `scripts/run_qwen38_days.sh`; when more
  than one GPU job runs at once use `lab-run --new --spare --gpus 1` (with `--cpus 24 --mem 110G`
  each for up to three parallel jobs). Give each command in its own `bash` block.
- The scripted partner (`coop.partner`) and its probe are a test harness; the experimental
  partner will be an LLM-simulated human (see `RESEARCH_STATUS.md` §6). Fix bugs in them, but
  don't spend effort refining their personas or probe labels.
- Keep runs comparable: compare conditions only within one output folder (`code_version.json`
  refuses resumes across code changes); a behaviour change means a new `--out`.
