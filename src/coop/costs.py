"""API cost per model and condition, for the reports (coop.learning, coop.analyze, coop.decision).

Costs come from the per-call ``cost_usd`` the API backends record (``coop.llm``), summed in
``llm_usage`` (ego), ``partner_llm.usage`` (partner), the knowledge probe and the day
feedback. The folder's ``cost_ledger.tsv`` holds the total of every call, the ground truth
(it also counts calls of episodes stopped by the budget and replayed on resume).
Old folders without API backends produce no lines.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ledger_total(out: Path) -> Optional[float]:
    p = Path(out) / "cost_ledger.tsv"
    if not p.exists():
        return None
    tot = 0.0
    for line in p.read_text().splitlines():
        try:
            tot += float(line.split("\t")[0])
        except ValueError:
            pass
    return tot


def _model_of(spec: Optional[Dict[str, Any]]) -> str:
    if not spec:
        return "?"
    from .llm import resolve_spec
    s = resolve_spec(spec)
    return f"{s.get('ref') or s.get('type', '')}:{s.get('model') or s.get('model_env') or '?'}"


def format_costs(rows: Iterable[Tuple[str, str, Dict[str, Any]]], ledger: Optional[float] = None) -> List[str]:
    """rows: (model, condition, usage dict with calls / tokens / cost_usd)."""
    acc: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for model, cond, u in rows:
        if not u or "cost_usd" not in u:
            continue
        a = acc[(model, cond)]
        for k in ("calls", "prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"):
            a[k] += float(u.get(k) or 0)
    if not acc and ledger is None:
        return []
    lines = ["API cost per model and condition",
             f"{'model':<32}{'condition':<22}{'calls':>8}{'prompt tok':>12}{'cached':>10}{'output tok':>12}"
             f"{'reasoning':>11}{'USD':>10}"]
    total = 0.0
    for (model, cond), a in sorted(acc.items()):
        total += a["cost_usd"]
        lines.append(f"{model:<32}{cond:<22}{a['calls']:>8.0f}{a['prompt_tokens']:>12.0f}{a['cached_tokens']:>10.0f}"
                     f"{a['completion_tokens']:>12.0f}{a['reasoning_tokens']:>11.0f}{a['cost_usd']:>10.2f}")
    lines.append(f"total in the logs: ${total:.2f}" + ("" if ledger is None else f"; cost ledger (every call, "
                                                         f"replayed ones included): ${ledger:.2f}"))
    return lines


def days_folder_costs(out: Path) -> List[str]:
    """Cost lines for a coop.days folder."""
    out = Path(out)
    cfg_path = out / "config.json"
    if not cfg_path.exists():
        return []
    from .days import DaysConfig
    cfg = DaysConfig(**json.loads(cfg_path.read_text()))
    ego_model = {c: _model_of(cfg.condition_agent(c).get("backend")) for c in cfg.conditions
                 if cfg.condition_agent(c).get("type") == "llm"}
    partner_model = _model_of(cfg.partner.get("backend")) if cfg.partner_type() == "llm" else None
    rows = []
    for f in sorted(out.glob("runs/*/*/seed*/day*/*/summary.json")):
        cond = f.parents[4].name
        s = json.loads(f.read_text())
        if cond in ego_model:
            rows.append((ego_model[cond], cond, s.get("llm_usage") or {}))
        pu = (s.get("partner_llm") or {}).get("usage")
        if partner_model and pu:
            rows.append((partner_model, f"{cond} (partner)", pu))
    for f in sorted(out.glob("runs/*/*/seed*/day*/knowledge_probe.json")):
        cond = f.parents[3].name
        d = json.loads(f.read_text())
        if cond in ego_model and "cost_usd" in d:
            rows.append((ego_model[cond], f"{cond} (probe)", {"calls": 1, **d}))
    for f in sorted(out.glob("runs/*/*/seed*/day*/feedback.json")):
        cond = f.parents[3].name
        d = json.loads(f.read_text())
        if partner_model and "cost_usd" in d:
            rows.append((partner_model, f"{cond} (feedback)", {"calls": 1, **d}))
    return format_costs(rows, ledger_total(out))
