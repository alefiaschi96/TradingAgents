"""Quick smoke test for the configured LLM models.

Sends ONE tiny prompt to each configured model (deep + quick) so you can catch a
bad model name, an auth failure, or an exhausted quota in a couple of seconds —
BEFORE spending a full 7-8 minute analysis on a broken config.

Usage:
    set -a && source .env.live && set +a
    python scripts/smoke_models.py
"""

from __future__ import annotations

import os
import sys
import time

# Make the repo root importable (live/, paper_sim/) when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.config import Config
from tradingagents.llm_clients import create_llm_client


def _ping(label: str, provider: str, model: str) -> bool:
    print(f"\n[{label}] provider={provider} model={model}")
    try:
        client = create_llm_client(provider=provider, model=model)
        llm = client.get_llm()
        t0 = time.time()
        resp = llm.invoke("Reply with the single word: OK")
        dt = time.time() - t0
        content = getattr(resp, "content", resp)
        print(f"  OK in {dt:.1f}s -- reply: {str(content)[:80]!r}")
        return True
    except Exception as e:  # noqa: BLE001 - we want to see ANY failure verbatim
        print(f"  FAIL -- {type(e).__name__}: {str(e)[:400]}")
        return False


def main() -> int:
    cfg = Config.from_env()
    print(f"provider={cfg.llm_provider} | deep={cfg.deep_model} | quick={cfg.quick_model}")
    ok_deep = _ping("DEEP ", cfg.llm_provider, cfg.deep_model)
    ok_quick = _ping("QUICK", cfg.llm_provider, cfg.quick_model)
    print()
    if ok_deep and ok_quick:
        print("==> Both models answer. Safe to launch the run.")
        return 0
    print("==> At least one model FAILED. Do NOT launch until both are green.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
