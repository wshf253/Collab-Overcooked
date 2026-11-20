"""LLM backends (stdlib only).

* ``openai_compat``: the OpenAI-compatible Chat Completions API (local vLLM). Ported from
  ICRL4AHT ``research/coop_tom/llm.py``.
* ``openai``: the OpenAI API (reasoning models: ``reasoning_effort``, ``max_completion_tokens``,
  no sampling parameters).
* ``anthropic``: the Anthropic Messages API (system prompt cached, effort via
  ``anthropic_params``).
* ``mock``: offline stand-in for pipeline tests.

The API types (``APIBackend``) add retries with ``Retry-After``, a concurrency limit shared by
every process on the machine, per-call cost from a price table, and a budget stop
(``BudgetExceeded``) driven by a cost ledger file (``COOP_COST_LEDGER`` / ``COOP_MAX_COST``).
API keys are read from environment variables only and are removed from every error text.
A backend spec may be ``{ref: NAME}``: the entry NAME of ``configs/coop/api/backends.yaml``.
No openai/anthropic/vllm/torch dependency.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


@dataclass
class LLMResult:
    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    finish_reason: str = ""
    # API backends (``extended``): prompt_tokens is the whole prompt, cached and written included
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    rate_limited_s: float = 0.0
    served_model: str = ""
    missing_usage: List[str] = field(default_factory=list)
    expects_reasoning: bool = False
    extended: bool = False

    @property
    def thinking(self) -> bool:
        """The server produced reasoning text despite thinking being requested off.
        An empty <think></think> block (some templates emit one when thinking is off) is not.
        Reasoning models (``expects_reasoning``) always reason: not a health problem."""
        if self.expects_reasoning:
            return False
        return bool(re.sub(r"</?think>", "", self.reasoning).strip())

    def api_fields(self) -> Dict[str, Any]:
        """The extra per-call fields of API backends (empty for the others, so old logs keep
        their shape)."""
        if not self.extended:
            return {}
        out = {"cached_tokens": self.cached_tokens, "cache_write_tokens": self.cache_write_tokens,
               "reasoning_tokens": self.reasoning_tokens, "cost_usd": round(self.cost_usd, 6),
               "retries": self.retries, "rate_limited_s": round(self.rate_limited_s, 2),
               "served_model": self.served_model}
        if self.missing_usage:
            out["missing_usage"] = list(self.missing_usage)
        return out


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    thinking: int = 0
    truncated: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    rate_limited_s: float = 0.0
    refusals: int = 0
    extended: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResult):
        with self.lock:
            self.calls += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.errors += bool(r.error)
            self.thinking += r.thinking
            self.truncated += r.finish_reason == "length"
            if r.extended:
                self.extended = True
                self.cached_tokens += r.cached_tokens
                self.cache_write_tokens += r.cache_write_tokens
                self.reasoning_tokens += r.reasoning_tokens
                self.cost_usd += r.cost_usd
                self.retries += r.retries
                self.rate_limited_s += r.rate_limited_s
                self.refusals += r.finish_reason == "refusal"

    def as_dict(self):
        out = {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
               "completion_tokens": self.completion_tokens, "errors": self.errors,
               "thinking": self.thinking, "truncated": self.truncated}
        if self.extended:      # API backends only: old outputs keep their keys
            out.update(cached_tokens=self.cached_tokens, cache_write_tokens=self.cache_write_tokens,
                       reasoning_tokens=self.reasoning_tokens, cost_usd=round(self.cost_usd, 6),
                       retries=self.retries, rate_limited_s=round(self.rate_limited_s, 2),
                       refusals=self.refusals)
        return out


class Backend:
    name = "base"

    def chat(self, messages: list[dict], max_tokens: Optional[int] = None, seed: Optional[int] = None,
             json_mode: Optional[bool] = None) -> LLMResult:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """POST /chat/completions. Thinking control goes in `chat_template_kwargs` (vLLM)."""

    def __init__(self, name: str, base_url: str, model: str, api_key_env: Optional[str] = None,
                 params: Optional[dict] = None, extra_body: Optional[dict] = None, json_mode: bool = True,
                 timeout_s: float = 120.0, max_retries: int = 4, send_seed: bool = True):
        self.name, self.model = name, model
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.params = dict(params or {})
        self.extra_body = dict(extra_body or {})
        self.json_mode, self.timeout_s = json_mode, timeout_s
        self.max_retries, self.send_seed = max_retries, send_seed
        # Cluster loopback/node addresses must not go through HTTP proxies.
        self.opener = build_opener(ProxyHandler({})) if _is_local(self.base_url) else build_opener()

    def payload(self, messages, max_tokens=None, seed=None, json_mode=None) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **self.params}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None and self.send_seed:
            body["seed"] = int(seed) % (2**31)
        if self.json_mode if json_mode is None else json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)  # raw REST: chat_template_kwargs, top_k, ... are top-level
        return body

    def post(self, endpoint: str, body: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = Request(f"{self.base_url}/{endpoint}", headers=headers,
                      data=None if body is None else json.dumps(body).encode())
        with self.opener.open(req, timeout=self.timeout_s) as resp:
            return json.load(resp)

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None) -> LLMResult:
        body = self.payload(messages, max_tokens, seed, json_mode)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.post("chat/completions", body)
                choice = resp["choices"][0]
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                m = _THINK_RE.search(text)
                if m:  # no reasoning parser on the server: strip inline thinking
                    reasoning = reasoning or m.group(0)
                    text = _THINK_RE.sub("", text).strip()
                u = resp.get("usage") or {}
                return LLMResult(text, reasoning, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0,
                                 time.time() - t0, finish_reason=choice.get("finish_reason") or "")
            except HTTPError as e:  # 4xx are config errors: do not retry
                last_err = f"HTTP {e.code}: {e.read(500).decode(errors='replace')}"
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except (URLError, TimeoutError, ValueError, KeyError, IndexError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        return LLMResult("", error=last_err)

    def models(self) -> list[str]:
        return [m["id"] for m in self.post("models").get("data", [])]


def _is_local(url: str) -> bool:
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    return host in ("localhost", "127.0.0.1") or host.startswith(("10.", "192.168.", "172."))


class MockBackend(Backend):
    """Offline stand-in for pipeline tests: well-formed JSON for every prompt, no reasoning.

    Decisions: picks from the ``VALID_ACTIONS:`` line that every decision prompt
    ends with, preferring progress (deliver > fill > start > put > pickup), seeded
    per call. Partner probes: a fixed neutral profile.
    """

    PRIORITY = ("deliver_soup", "fill_dish_with_food", "cut(", "cook(", "bake(", "stir(",
                "put_obj_in_utensil", "pickup(", "place_obj_on_counter")

    def __init__(self, name="mock", seed: int = 0):
        self.name = name
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.calls: list = []

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None) -> LLMResult:
        with self._lock:
            self.calls.append(messages)
        rng = random.Random(seed) if seed is not None else self._rng
        text = "\n".join(m["content"] for m in messages)
        last = messages[-1]["content"]
        if "PARTNER_PROFILE_PROBE" in last:
            out = {"partner_profile": {"prefers": "no clear preference", "avoids": "nothing",
                                       "gives_way": "does not give way", "speed": "normal",
                                       "reliability": "reliable", "commitment": "finishes tasks",
                                       "orders": "no fixed habit"},
                   "stage_shares": {"prep": 34, "cook": 33, "serve": 33},
                   "notes": "mock notes"}
        elif "EPISODE_NOTE" in last:
            out = {"note": "mock note: I took the oldest order, my partner took the other."}
        elif "PARTNER_FILE_UPDATE" in last:
            out = {"partner_file": "## Partner Preferences\n- When both pots are free, they use pot0. Evidence: +D1E1. "
                                   "Source: observed\n\n## Partner Reactions\n- (nothing yet)\n\n## Cooperation "
                                   "Strategy\n- I use pot1. Follows from: pot0 preference\n\n## Uncertainties\n"
                                   "- Do they prefer an order?; watch their first pick"}
        elif "DAY_FEEDBACK" in last:
            out = {"feedback": "Good day. Please leave the oven to me."}
        elif "KNOWLEDGE_PROBE" in last:
            ids = re.findall(r"^([\w:]+): ", last, flags=re.M)
            out = {"answers": {i: "no preference" for i in ids}}
        elif "PARTNER_MODEL_UPDATE" in last:
            out = {"partner_model": "## How they choose what to do (with evidence)\n- mock habit\n\n"
                                    "## How they react to me\n- mock\n\n## What I should do with them\n- mock"
                                    "\n\n## Open questions\n- mock"}
        else:
            valid = [a for a in _grab_list(last, "VALID_ACTIONS:") if not a.startswith("wait")]
            act = "wait(2)"
            for p in self.PRIORITY:
                hits = [a for a in valid if a.startswith(p)]
                if hits:
                    act = rng.choice(hits)
                    break
            out = {}
            if '"reconsider_when"' in text:      # LLM persona partner: intention state
                opened = re.search(r"Open orders \(oldest first\): ([\w_]+)", last)
                out.update({"goal": "mock goal", "order": opened.group(1) if opened else "none",
                            "reconsider_when": "mock"})
            elif '"doing"' in text:
                opened = re.search(r"Open orders \(oldest first\): ([\w_]+)", last)
                out["partner"] = {"doing": "mock", "order": opened.group(1) if opened else "unknown",
                                  "next": "pickup(dish, dish_dispenser)", "habit": "mock habit"}
            elif '"partner"' in text:
                out["partner"] = "mock belief"
            out["action"] = act
        s = json.dumps(out)
        return LLMResult(s, prompt_tokens=len(text) // 4, completion_tokens=len(s) // 4, finish_reason="stop")


def _grab_list(text: str, tag: str) -> list:
    for line in reversed(text.splitlines()):
        if line.strip().startswith(tag):
            return [x.strip() for x in line.split(tag, 1)[1].split("|") if x.strip()]
    return []


# ---------------------------------------------------------------------- #
# API backends (OpenAI, Anthropic): shared machinery
# ---------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs" / "coop" / "api" / "backends.yaml"
PRICES = REPO / "configs" / "coop" / "api" / "prices.yaml"
RETRY_CODES = (429, 500, 502, 503, 504, 529)
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed")


class BudgetExceeded(RuntimeError):
    """The cost ledger reached ``COOP_MAX_COST``: no new API call starts."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def registry(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    return _load_yaml(Path(path or os.environ.get("COOP_BACKENDS") or REGISTRY))


def price_table(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    return _load_yaml(Path(path or os.environ.get("COOP_PRICES") or PRICES))


def resolve_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``{ref: NAME, ...overrides}`` -> the registry entry NAME with the overrides on top
    (``ref`` kept, so outputs record which entry was used). Other specs unchanged."""
    if not spec or "ref" not in spec:
        return spec
    reg = registry()
    if spec["ref"] not in reg:
        raise ValueError(f"backend ref {spec['ref']!r} not in {REGISTRY} (known: {sorted(reg)})")
    out = copy.deepcopy(reg[spec["ref"]])
    out.update({k: v for k, v in spec.items() if k != "ref"})
    out["ref"] = spec["ref"]
    return out


class SlotLimiter:
    """At most ``n`` concurrent holders across every process on this machine: one lock file
    per slot (``flock``), released when the holder closes it or its process dies."""

    def __init__(self, name: str, n: int):
        import fcntl  # noqa: F401  (POSIX only; the lab server and CI are Linux)
        self.n = int(n)
        root = Path(os.environ.get("COOP_LOCK_DIR") or Path(tempfile.gettempdir()) / "coop_api_slots")
        root.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.-]", "_", name)
        self.paths = [root / f"{safe}.{i}.lock" for i in range(self.n)]
        self._local = threading.local()

    def __enter__(self):
        import fcntl
        while True:
            for p in random.sample(self.paths, len(self.paths)):
                fh = open(p, "a")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._local.fh = fh
                    return self
                except OSError:
                    fh.close()
            time.sleep(0.05 + 0.1 * random.random())

    def __exit__(self, *exc):
        import fcntl
        fh = getattr(self._local, "fh", None)
        if fh is not None:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
            self._local.fh = None
        return False


class CostLedger:
    """Cost of every API call, appended to one file shared by all processes of a run
    (``COOP_COST_LEDGER``); ``COOP_MAX_COST`` stops new calls once the total reaches it.
    Calls already running when the limit is reached finish, so the total can overshoot by
    about ``max_concurrency`` calls."""
    _state: Dict[str, List[float]] = {}          # path -> [offset, total] (per process)
    _lock = threading.Lock()

    @classmethod
    def path(cls) -> Optional[str]:
        return os.environ.get("COOP_COST_LEDGER") or None

    @classmethod
    def total(cls) -> float:
        p = cls.path()
        if not p or not os.path.exists(p):
            return 0.0
        with cls._lock:
            off, tot = cls._state.get(p, [0, 0.0])
            with open(p) as fh:
                fh.seek(off)
                for line in fh:
                    if line.endswith("\n"):
                        try:
                            tot += float(line.split("\t")[0])
                        except ValueError:
                            pass
                        off += len(line.encode())
            cls._state[p] = [off, tot]
            return tot

    @classmethod
    def check(cls) -> None:
        limit = os.environ.get("COOP_MAX_COST")
        if limit and cls.total() >= float(limit):
            raise BudgetExceeded(f"cost ledger {cls.path()} reached the limit ${float(limit):.2f}")

    @classmethod
    def record(cls, cost: float, label: str) -> None:
        p = cls.path()
        if not p or cost <= 0:
            return
        import fcntl
        with open(p, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{cost:.8f}\t{label}\n")
            fcntl.flock(fh, fcntl.LOCK_UN)


def call_cost(price: Optional[Dict[str, Any]], prompt: int, cached: int, written: int, output: int,
              reasoning: int) -> float:
    """USD for one call. ``prompt`` is the whole prompt; uncached = prompt - cached - written.
    ``output`` includes reasoning unless the price entry says ``reasoning_in_output: false``."""
    if not price:
        return 0.0
    uncached = max(0, prompt - cached - written)
    out = output + (0 if price.get("reasoning_in_output", True) else reasoning)
    return (uncached * float(price.get("input", 0)) + cached * float(price.get("cached_input", price.get("input", 0)))
            + written * float(price.get("cache_write", price.get("input", 0))) + out * float(price.get("output", 0))) / 1e6


class APIError(RuntimeError):
    pass


class APIBackend(Backend):
    """Base of the hosted-API backends: keys from the environment, retries, slots, cost."""

    def __init__(self, name: str, base_url: str, model: str, api_key_env: Optional[str] = None,
                 timeout_s: float = 300.0, max_retries: int = 6, backoff_s: float = 2.0,
                 max_concurrency: Optional[int] = None, reasoning_allowance: Optional[int] = None,
                 expects_reasoning: bool = False, price: Optional[Dict[str, Any]] = None,
                 slot_name: Optional[str] = None, send_sampling: bool = False,
                 params: Optional[dict] = None, extra_body: Optional[dict] = None, json_mode: bool = True):
        self.name, self.model = name, model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.timeout_s, self.max_retries, self.backoff_s = float(timeout_s), int(max_retries), float(backoff_s)
        self.expects_reasoning = bool(expects_reasoning)
        self.reasoning_allowance = int(reasoning_allowance if reasoning_allowance is not None
                                       else (3000 if expects_reasoning else 0))
        self.price = dict(price) if price else None
        self.send_sampling = bool(send_sampling)
        self.params = {k: v for k, v in (params or {}).items() if send_sampling or k not in SAMPLING_KEYS}
        self.extra_body = dict(extra_body or {})
        self.json_mode = json_mode
        self.limiter = SlotLimiter(slot_name or f"{type(self).__name__}_{model}", max_concurrency) \
            if max_concurrency else None
        self.opener = build_opener(ProxyHandler({})) if _is_local(self.base_url) else build_opener()

    def __repr__(self):            # never show the key
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r})"

    def redact(self, text: str) -> str:
        if self._key:
            text = text.replace(self._key, "[redacted]")
        return re.sub(r"(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]+", r"\1[redacted]", text)

    def cap(self, max_tokens: Optional[int]) -> int:
        return int(max_tokens if max_tokens is not None else 1024) + self.reasoning_allowance

    # -- transport --------------------------------------------------------
    def headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def send(self, endpoint: str, body: Dict[str, Any]) -> tuple:
        """(response json, retries, seconds waited on 429). Raises APIError (text redacted)."""
        waited, last = 0.0, ""
        for attempt in range(self.max_retries + 1):
            req = Request(f"{self.base_url}/{endpoint}", headers=self.headers(), data=json.dumps(body).encode())
            try:
                with self.opener.open(req, timeout=self.timeout_s) as resp:
                    return json.load(resp), attempt, waited
            except HTTPError as e:
                text = self.redact(e.read(2000).decode(errors="replace"))
                last = f"HTTP {e.code}: {text}"
                retry_after = e.headers.get("retry-after") if e.headers else None
                spend_cap = e.code == 429 and "enforced_spend_limit_reached" in text
                if e.code not in RETRY_CODES or spend_cap or attempt == self.max_retries:
                    raise APIError(last) from None
                try:
                    wait = float(retry_after) if retry_after is not None else None
                except ValueError:
                    wait = None
                if wait is None:
                    wait = min(60.0, self.backoff_s * 2 ** attempt + random.random() * self.backoff_s)
                if e.code == 429:
                    waited += wait
            except (URLError, TimeoutError, ConnectionError, ValueError) as e:
                last = self.redact(f"{type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    raise APIError(last) from None
                wait = min(60.0, self.backoff_s * 2 ** attempt + random.random() * self.backoff_s)
            time.sleep(wait)
        raise APIError(last)

    # -- one call -----------------------------------------------------------
    def payload(self, messages, max_tokens=None, seed=None, json_mode=None) -> Dict[str, Any]:
        raise NotImplementedError

    def parse(self, resp: Dict[str, Any]) -> LLMResult:
        raise NotImplementedError

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None) -> LLMResult:
        CostLedger.check()
        body = self.payload(messages, max_tokens, seed, json_mode)
        t0 = time.time()
        try:
            if self.limiter is not None:
                with self.limiter:
                    resp, retries, waited = self.send(self.endpoint, body)
            else:
                resp, retries, waited = self.send(self.endpoint, body)
            r = self.parse(resp)
        except APIError as e:
            return LLMResult("", error=str(e), latency_s=time.time() - t0, extended=True,
                             expects_reasoning=self.expects_reasoning)
        except (KeyError, IndexError, TypeError) as e:
            return LLMResult("", error=self.redact(f"unexpected response shape: {type(e).__name__}: {e}"),
                             latency_s=time.time() - t0, extended=True, expects_reasoning=self.expects_reasoning)
        r.latency_s, r.retries, r.rate_limited_s = time.time() - t0, retries, waited
        r.extended, r.expects_reasoning = True, self.expects_reasoning
        r.cost_usd = call_cost(self.price, r.prompt_tokens, r.cached_tokens, r.cache_write_tokens,
                               r.completion_tokens, r.reasoning_tokens)
        CostLedger.record(r.cost_usd, self.name)
        return r


class OpenAIBackend(APIBackend):
    """The OpenAI Chat Completions API for reasoning models: ``reasoning_effort``,
    ``max_completion_tokens`` (caller cap + reasoning allowance), no temperature / top_p /
    seed unless ``send_sampling``. ``prompt_cache_key: auto`` sends a key derived from the
    system prompt (the stable prefix)."""
    endpoint = "chat/completions"

    def __init__(self, *a, reasoning_effort: Optional[str] = None, prompt_cache_key: Optional[str] = None, **kw):
        super().__init__(*a, **kw)
        self.reasoning_effort, self.prompt_cache_key = reasoning_effort, prompt_cache_key

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self._key:
            h["Authorization"] = "Bearer " + self._key
        return h

    def payload(self, messages, max_tokens=None, seed=None, json_mode=None):
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **self.params,
                                "max_completion_tokens": self.cap(max_tokens)}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if seed is not None and self.send_sampling:
            body["seed"] = int(seed) % (2**31)
        if self.json_mode if json_mode is None else json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.prompt_cache_key:
            if self.prompt_cache_key == "auto":
                import hashlib
                sys_text = "".join(m["content"] for m in messages if m.get("role") == "system")
                body["prompt_cache_key"] = "coop-" + hashlib.sha256(sys_text.encode()).hexdigest()[:24]
            else:
                body["prompt_cache_key"] = self.prompt_cache_key
        body.update(self.extra_body)
        return body

    def parse(self, resp):
        choice = resp["choices"][0]
        msg = choice.get("message") or {}
        u = resp.get("usage") or {}
        missing = [k for k in ("prompt_tokens", "completion_tokens") if k not in u]
        pd, cd = u.get("prompt_tokens_details"), u.get("completion_tokens_details")
        if not isinstance(pd, dict) or "cached_tokens" not in pd:
            missing.append("prompt_tokens_details.cached_tokens")
        if not isinstance(cd, dict) or "reasoning_tokens" not in cd:
            missing.append("completion_tokens_details.reasoning_tokens")
        fr = choice.get("finish_reason") or ""
        return LLMResult(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or "",
                         int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                         finish_reason="refusal" if msg.get("refusal") else fr,
                         cached_tokens=int((pd or {}).get("cached_tokens") or 0),
                         reasoning_tokens=int((cd or {}).get("reasoning_tokens") or 0),
                         served_model=resp.get("model") or "", missing_usage=missing)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


class AnthropicBackend(APIBackend):
    """The Anthropic Messages API (``POST /v1/messages``). The system message becomes the
    top-level ``system`` field as one text block marked for caching; ``max_tokens`` (required)
    is the caller cap + the reasoning allowance; ``anthropic_params`` go into the body as given
    (e.g. ``{output_config: {effort: low}}``). No sampling parameters unless ``send_sampling``
    (API version 2026-10-21 rejects them). No JSON response format: the prompt and
    ``parse_json`` handle it."""
    endpoint = "messages"

    def __init__(self, *a, anthropic_version: str = "2026-10-21", anthropic_params: Optional[dict] = None,
                 anthropic_beta: Optional[List[str]] = None, cache_system: bool = True, **kw):
        super().__init__(*a, **kw)
        self.anthropic_version = anthropic_version
        self.anthropic_params = dict(anthropic_params or {})
        self.anthropic_beta = list(anthropic_beta or [])
        self.cache_system = cache_system

    def headers(self):
        h = {"Content-Type": "application/json", "anthropic-version": self.anthropic_version}
        if self._key:
            h["x-api-key"] = self._key
        if self.anthropic_beta:
            h["anthropic-beta"] = ",".join(self.anthropic_beta)
        return h

    def payload(self, messages, max_tokens=None, seed=None, json_mode=None):
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        rest = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
        body: Dict[str, Any] = {"model": self.model, "max_tokens": self.cap(max_tokens), "messages": rest, **self.params}
        if system:
            block: Dict[str, Any] = {"type": "text", "text": system}
            if self.cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            body["system"] = [block]
        body = _deep_merge(body, self.anthropic_params)
        body.update(self.extra_body)
        return body

    STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "refusal"}

    def parse(self, resp):
        blocks = resp.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        reasoning = "".join(b.get("thinking", "") or "" for b in blocks if b.get("type") == "thinking")
        u = resp.get("usage") or {}
        missing = [k for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                               "cache_creation_input_tokens") if k not in u]
        od = u.get("output_tokens_details")
        if not isinstance(od, dict) or "thinking_tokens" not in od:
            missing.append("output_tokens_details.thinking_tokens")
        read = int(u.get("cache_read_input_tokens") or 0)
        write = int(u.get("cache_creation_input_tokens") or 0)
        stop = resp.get("stop_reason") or ""
        return LLMResult(text, reasoning, int(u.get("input_tokens") or 0) + read + write,
                         int(u.get("output_tokens") or 0), finish_reason=self.STOP.get(stop, stop),
                         cached_tokens=read, cache_write_tokens=write,
                         reasoning_tokens=int((od or {}).get("thinking_tokens") or 0),
                         served_model=resp.get("model") or "", missing_usage=missing)


API_TYPES = {"openai": OpenAIBackend, "anthropic": AnthropicBackend}
API_DEFAULT_URL = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1"}


def build_api_backend(spec: Dict[str, Any], name: str) -> APIBackend:
    kind = spec["type"]
    model = (os.environ.get(spec["model_env"], "") if spec.get("model_env") else "") or spec.get("model", "")
    if not model:
        raise ValueError(f"backend '{name}': set model")
    base_url = (os.environ.get(spec["base_url_env"], "") if spec.get("base_url_env") else "") \
        or spec.get("base_url") or API_DEFAULT_URL[kind]
    price = spec.get("price")
    if isinstance(price, str):
        table = price_table()
        if price not in table:
            raise ValueError(f"backend '{name}': price {price!r} not in {PRICES}")
        price = table[price]
    kw = dict(api_key_env=spec.get("api_key_env"), timeout_s=spec.get("timeout_s", 300.0),
              max_retries=spec.get("max_retries", 6), backoff_s=spec.get("backoff_s", 2.0),
              max_concurrency=spec.get("max_concurrency"), reasoning_allowance=spec.get("reasoning_allowance"),
              expects_reasoning=spec.get("expects_reasoning", False), price=price,
              slot_name=spec.get("ref") or spec.get("slot_name"), send_sampling=spec.get("send_sampling", False),
              params=spec.get("params"), extra_body=spec.get("extra_body"),
              json_mode=spec.get("json_mode", kind == "openai"))
    if kind == "openai":
        return OpenAIBackend(name, base_url, model, reasoning_effort=spec.get("reasoning_effort"),
                             prompt_cache_key=spec.get("prompt_cache_key"), **kw)
    return AnthropicBackend(name, base_url, model, anthropic_version=spec.get("anthropic_version", "2026-10-21"),
                            anthropic_params=spec.get("anthropic_params"), anthropic_beta=spec.get("anthropic_beta"),
                            cache_system=spec.get("cache_system", True), **kw)


def build_backend(spec: dict, name: str = "backend") -> Backend:
    """Backend from a config dict. ``*_env`` keys read the value from an environment variable;
    ``ref`` names an entry of the backend registry."""
    spec = resolve_spec(spec)
    kind = spec.get("type", "openai_compat")
    if kind in API_TYPES:
        return build_api_backend(spec, name)
    if kind == "mock":
        return MockBackend(name, seed=spec.get("seed", 0))
    if kind == "openai_compat":
        base_url = os.environ.get(spec["base_url_env"], "") if spec.get("base_url_env") else ""
        base_url = base_url or spec.get("base_url", "")
        if not base_url:
            raise ValueError(f"backend '{name}': set base_url or the env var {spec.get('base_url_env')}")
        extra = dict(spec.get("extra_body") or {})
        env_kwargs = os.environ.get(spec["chat_template_kwargs_env"], "") if spec.get("chat_template_kwargs_env") else ""
        if env_kwargs:  # e.g. set by the serving script after its thinking-off check
            extra["chat_template_kwargs"] = json.loads(env_kwargs)
        model = (os.environ.get(spec["model_env"], "") if spec.get("model_env") else "") or spec.get("model", "")
        if not model:
            raise ValueError(f"backend '{name}': set model or the env var {spec.get('model_env')}")
        return OpenAICompatBackend(
            name=name, base_url=base_url, model=model, api_key_env=spec.get("api_key_env"),
            params=spec.get("params"), extra_body=extra, json_mode=spec.get("json_mode", True),
            timeout_s=spec.get("timeout_s", 120.0), max_retries=spec.get("max_retries", 4),
            send_seed=spec.get("send_seed", True))
    raise ValueError(f"unknown backend type {kind}")


def parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction (handles code fences / leading prose)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        # strict=False: allow raw line breaks inside strings (common in long markdown values)
        v = json.loads(t, strict=False)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    depth, start = 0, None
    for k, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    v = json.loads(t[start:k + 1], strict=False)
                    if isinstance(v, dict):
                        return v
                except Exception:
                    start = None
    return None
