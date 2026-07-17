"""Browser dashboard for the paper-sim, in plain language. Run it, open the link:

    python -m paper_sim.dashboard            # http://localhost:8765
    python -m paper_sim.dashboard 9000       # custom port

Pure standard library (http.server) — no pip installs. The page auto-refreshes
and explains in plain Italian what the bot is doing, so anyone can follow it.
Reads the same state.json + the current run's log the simulator writes. Ctrl-C to stop.
Bound to localhost only (not exposed to the network).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live.run_logging import default_text_log

_STATE = os.path.expanduser(
    os.environ.get("PAPER_STATE_PATH", "~/.tradingagents/paper_sim/state.json")
)
def _resolve_log() -> str:
    """Resolve the current run's text log on EACH call, not once at import.

    The dashboard is started just before the daemon, so latest.log may not exist
    yet at startup; resolving per request lets it pick the log up as soon as the
    daemon creates it (otherwise it sticks to the fallback path forever)."""
    return os.environ.get("PAPER_LOG") or default_text_log("paper_sim")
_STALE_MIN = float(os.environ.get("WATCH_STALE_MIN", "12"))


def _port_from_argv() -> int:
    """Optional first CLI arg is the port; ignore non-numeric argv (e.g. when
    the module is imported under pytest) so import never crashes."""
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    return int(os.environ.get("DASHBOARD_PORT", "8765"))


_PORT = _port_from_argv()

_KEY = (
    "paper-sim started", "regime ready", "regime not ready", "setup detected",
    "Analysis decision for", "VETOED", "regime gate OK", "no entry",
    "OPEN PF", "CLOSE ", "exceeded", "loop error", "market gate short-circuit",
    "regime transition",
)


def _tail_lines(path: str, nbytes: int = 131072) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return []


def _last_ts(lines: list[str]) -> datetime | None:
    for line in reversed(lines):
        if len(line) >= 19 and line[:4].isdigit() and line[4] == "-":
            try:
                return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


def _msg(ln: str) -> str:
    return (ln.split(" | ", 1)[1] if " | " in ln else ln).strip()


def _state_ts(iso: str) -> str:
    """Format a state timestamp as 'DD/MM HH:MM:SS' in LOCAL time.

    State timestamps are UTC (_now_iso) while the text log is local time, so
    without converting, the closed-trades table was off by the UTC offset versus
    the 'Ultime mosse' list (e.g. 12:02 vs 14:02 in CEST)."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d/%m %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso)[11:19]


def _veto_message(low: str) -> str:
    """Spell out *why* a trade was vetoed, one clear Italian line per case.

    The two gates log ``... VETOED by <analyst|regime> gate: <reason>``. We key
    off the gate and the reason so every veto is explicit in the feed instead of
    a single catch-all. Analyst gate: no data / HOLD / opposite view. Regime
    gate: chop / against the higher-timeframe trend / overextended. Unknown or
    older unlabelled vetoes fall back to a generic line."""
    reason = low.split("gate:", 1)[1].strip() if "gate:" in low else low
    if "analyst gate" in low:
        if "no reliable" in reason:
            return "Operazione annullata: l'analista di mercato non aveva dati affidabili"
        if "hold/flat" in reason or "no directional" in reason:
            return "Operazione annullata: l'analista di mercato consigliava di ASPETTARE"
        if "against market analyst" in reason:
            return "Operazione annullata: andava CONTRO il parere dell'analista di mercato"
        return "Operazione annullata dall'analista di mercato"
    # regime gate (or older/unlabelled vetoes)
    if "chop" in reason:
        return "Operazione annullata: mercato LATERALE, nessuna direzione chiara"
    if "against htf" in reason:
        return "Operazione annullata: andava CONTRO l'andamento di fondo"
    if "overextended" in reason or "chasing" in reason:
        return "Operazione annullata: prezzo troppo lontano dalla media (inseguirebbe)"
    return "Operazione annullata (contro l'andamento)"


def _humanize(lines: list[str]) -> list[dict]:
    """Turn key log lines into plain-Italian one-liners with a colour 'kind'."""
    out: list[dict] = []
    for ln in lines:
        m = _msg(ln)
        low = m.lower()
        t = (ln[8:10] + "/" + ln[5:7] + " " + ln[11:19]) if len(ln) >= 19 and ln[4:5] == "-" else ""
        h = k = None
        if "monitor open" in low:
            continue
        if "analysis decision" in low:
            r = m.split(":")[-1].strip().lower()
            h = {
                "buy": "Gli analisti vogliono COMPRARE",
                "overweight": "Gli analisti vogliono COMPRARE (cauti)",
                "hold": "Gli analisti dicono ASPETTARE",
                "underweight": "Gli analisti vogliono VENDERE (cauti)",
                "sell": "Gli analisti vogliono VENDERE",
            }.get(r, m)
            k = "dec"
        elif "market gate short-circuit" in low:
            # The graph stopped right after the market analyst: no debate/PM
            # was run at all (cost saving), which is different from a veto of a
            # completed decision — say so explicitly.
            h = (
                "Analisi fermata subito: l'analista non aveva dati affidabili (risparmio)"
                if "no reliable" in low
                else "Analisi fermata subito: l'analista consiglia di ASPETTARE (risparmio)"
            )
            k = "veto"
        elif "vetoed" in low:
            h, k = _veto_message(low), "veto"
        elif "regime gate ok" in low:
            h, k = "operazione approvata", "ok"
        elif "no entry" in low:
            h, k = "nessuna mossa, resta fermo", "muted"
        elif m.startswith("OPEN ") and " sell " in (" " + low + " "):
            h, k = "Aperta una scommessa al RIBASSO (punta che scende)", "open"
        elif m.startswith("OPEN "):
            h, k = "Aperta una scommessa al RIALZO (punta che sale)", "open"
        elif m.startswith("CLOSE ") and "via tp" in low:
            h, k = "Chiusa un'operazione in GUADAGNO", "win"
        elif m.startswith("CLOSE ") and "via sl" in low:
            h, k = "Chiusa un'operazione in PERDITA", "loss"
        elif "regime transition" in low:
            h, k = "Cambio di andamento appena nato: analisi anticipata", "think"
        elif "regime not ready" in low:
            h, k = "Mercato indeciso: il bot resta fermo", "muted"
        elif "regime ready" in low:
            h, k = "Andamento chiaro: il bot si mette ad analizzare", "muted"
        elif "setup detected" in low:
            h, k = "Sta studiando il mercato...", "think"
        elif "started" in low:
            h, k = "Bot avviato", "muted"
        elif "exceeded" in low:
            h, k = "Analisi troppo lenta: la riprovo", "warn"
        elif "loop error" in low:
            h, k = "Intoppo temporaneo, vado avanti", "warn"
        if h:
            out.append({"t": t, "h": h, "k": k})
    return out[-9:]


def _current_price(lines: list[str]) -> float | None:
    for ln in reversed(lines):
        if "monitor open" in ln and "[" in ln and "]" in ln:
            try:
                a, b = (float(x) for x in ln[ln.index("[") + 1: ln.index("]")].split(","))
                return (a + b) / 2.0
            except (ValueError, IndexError):
                return None
    return None


def _price_path(lines: list[str], entry: float) -> list[float]:
    """Price path of the *current* trade: entry, then the mid of each 1m monitor
    candle logged since the last OPEN. Built from the bot's own log — no network."""
    start = 0
    for i, ln in enumerate(lines):
        if "OPEN PF" in ln:
            start = i
    pts = [entry]
    for ln in lines[start:]:
        if "monitor open" in ln and "[" in ln and "]" in ln:
            try:
                a, b = (float(x) for x in ln[ln.index("[") + 1: ln.index("]")].split(","))
                pts.append((a + b) / 2.0)
            except (ValueError, IndexError):
                pass
    return pts[-120:]


def _asset_name() -> str:
    """Display name of the traded asset, from the env (PF_ADAUSD -> ADA).

    Hardcoding "SOL" was wrong once we run multiple instruments; the launcher
    exports SYMBOL per instance, so read it from there."""
    import re
    sym = (os.environ.get("ANALYSIS_SYMBOL") or os.environ.get("SYMBOL") or "").upper()
    s = re.sub(r"^PF_", "", sym)
    s = re.split(r"[-/]", s)[0]
    s = re.sub(r"USD$", "", s)
    return s or "il prezzo"


def _fmt_price(x: float) -> str:
    """Price with decimals adapted to magnitude (SOL ~73 -> 3dp, ADA ~0.16 -> 4dp),
    so a low-priced asset doesn't collapse to '0.16'."""
    a = abs(x)
    d = 2 if a >= 100 else 3 if a >= 1 else 4 if a >= 0.01 else 6
    return f"{x:.{d}f}"


def _bet(op: dict | None, lines: list[str]) -> dict | None:
    if not op:
        return None
    up = op.get("side") == "buy"
    entry, sl, tp = float(op.get("entry", 0)), float(op.get("sl", 0)), float(op.get("tp", 0))
    now = _current_price(lines)
    winning = None if now is None else (now > entry if up else now < entry)
    asset = _asset_name()
    return {
        "up": up,
        "dir_h": f"Punta che {asset} SALE" if up else f"Punta che {asset} SCENDE",
        "entry": entry, "stop": sl, "target": tp, "now": now, "winning": winning,
        "path": _price_path(lines, entry),
        "plain": (
            f"Ha {'comprato' if up else 'venduto'} a {_fmt_price(entry)}. "
            f"Chiude in GUADAGNO se {asset} {'sale' if up else 'scende'} a {_fmt_price(tp)}; "
            f"in PERDITA se {'scende' if up else 'sale'} a {_fmt_price(sl)}."
        ),
    }


def _snapshot() -> dict:
    lines = _tail_lines(_resolve_log())
    ts = _last_ts(lines)
    stale = False
    age = last_log = None
    if ts is not None:
        age = (datetime.now() - ts).total_seconds() / 60.0
        last_log = ts.strftime("%d/%m %H:%M:%S")
        stale = age >= _STALE_MIN

    s = None
    if os.path.exists(_STATE):
        try:
            s = json.load(open(_STATE))
        except (json.JSONDecodeError, OSError):
            s = None

    snap: dict = {"stale": stale, "age_min": age, "last_log": last_log,
                  "events": _humanize([ln for ln in lines if any(k in ln for k in _KEY)])}

    if s:
        eq = float(s.get("equity", 0.0))
        pnl = float(s.get("pnl_total", 0.0))
        w, l = int(s.get("wins", 0)), int(s.get("losses", 0))
        n = w + l
        start = eq - pnl
        closed = s.get("closed", [])
        snap.update(
            start=start, equity=eq, pnl=pnl,
            ret=(pnl / start * 100.0) if start else 0.0,
            up=pnl >= 0, trades=n, wins=w, losses=l,
            winrate=(w / n * 100.0) if n else 0.0,
            equity_curve=[start] + [float(t.get("equity_after", eq)) for t in closed],
            bet=_bet(s.get("open"), lines),
            closed=[
                {
                    "at": _state_ts(t.get("closed_at", "")),
                    "up": t.get("side") == "buy",
                    "win": (t.get("pnl", 0) or 0) >= 0,
                    "pnl": t.get("pnl"),
                }
                for t in closed[-8:]
            ],
        )
    else:
        snap.update(start=None, equity=None, pnl=0, ret=0, up=True, trades=0,
                    wins=0, losses=0, winrate=0, equity_curve=[], bet=None, closed=[])

    if stale:
        snap["head"] = {"text": "Il bot sembra BLOCCATO", "kind": "bad"}
    elif snap["bet"]:
        snap["head"] = {"text": "C'è una scommessa aperta — " + snap["bet"]["dir_h"], "kind": "open"}
    elif any(x in " ".join(lines[-6:]).lower()
             for x in ("generativelanguage", "afc is enabled", "setup detected")):
        snap["head"] = {"text": "Sta studiando il mercato per decidere…", "kind": "think"}
    else:
        snap["head"] = {"text": "Nessuna scommessa aperta — aspetta il momento giusto", "kind": "idle"}
    return snap


# --- Reasoning page (separate from the main dashboard) --------------------
_REASON_FIELDS = [
    ("Market analyst (tecnica intraday)", "market_report"),
    ("News / catalizzatori", "news_report"),
    ("Toro — tesi rialzista", "bull_case"),
    ("Orso — tesi ribassista", "bear_case"),
    ("Research manager", "research_manager_plan"),
    ("Trader (proposta)", "trader_proposal"),
    ("Rischio · aggressivo", "risk_aggressive"),
    ("Rischio · conservativo", "risk_conservative"),
    ("Rischio · neutrale", "risk_neutral"),
    ("Portfolio Manager — decisione finale", "pm_decision"),
]


def _log_dir() -> str:
    return os.path.dirname(os.path.abspath(_resolve_log()))


def _reasoning_records(limit: int = 30) -> list[dict]:
    """Collect the ``analysis`` events (full agent reasoning) across ALL run
    files in the log dir, most recent first.

    Reading every run's JSONL (not just the current one) means the reasoning
    history survives restarts — each restart writes a new file, and the analysis
    behind the currently-open trade often lives in an earlier one.
    """
    recs: list[dict] = []
    for path in glob.glob(os.path.join(_log_dir(), "*.jsonl")):
        if os.path.islink(path):  # skip latest.jsonl so we don't double-count
            continue
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("event") != "analysis":
                        continue
                    ts = str(r.get("ts", ""))
                    recs.append({
                        "iso": ts,  # ISO sorts chronologically as a string
                        "date": ts[:10],
                        "ts": ts[11:19],
                        "rating": r.get("rating"),
                        "sections": [
                            {"label": label, "text": (r.get(key) or "").strip()}
                            for label, key in _REASON_FIELDS
                        ],
                    })
        except OSError:
            continue
    recs.sort(key=lambda x: x["iso"], reverse=True)
    return recs[:limit]


_PAGE = r"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Il mio bot · come va</title>
<style>
:root{
 --bg:#070b12;--card-a:#131c2b;--card-b:#0e1622;--line:#1f2a3b;--line2:#2b3a52;
 --txt:#eaf1fb;--dim:#8696ad;--green:#34d399;--red:#fb7185;--blue:#60a5fa;
 --amber:#fbbf24;--accent:#8aa2ff}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;color:var(--txt);-webkit-font-smoothing:antialiased;
 font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 background:radial-gradient(1100px 520px at 50% -8%,#16223e 0%,#0a101c 55%,var(--bg) 100%);
 background-attachment:fixed;height:100dvh;overflow:hidden;padding:12px;display:flex}
.app{width:100%;max-width:1300px;margin:auto;height:100%;display:grid;
 grid-template-rows:auto minmax(0,1fr);gap:10px;min-height:0}
.main{display:grid;grid-template-columns:1.05fr .95fr;grid-template-rows:minmax(0,1fr);gap:10px;min-height:0}
.col{display:grid;gap:10px;min-height:0}
.col.left{grid-template-rows:auto minmax(0,1fr)}
.col.right{grid-template-rows:minmax(0,1fr) minmax(0,1fr)}
.card{background:linear-gradient(180deg,var(--card-a),var(--card-b));
 border:1px solid var(--line);border-radius:14px;
 box-shadow:0 1px 0 rgba(255,255,255,.04) inset,0 16px 36px -28px rgba(0,0,0,.9)}
.micro{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);font-weight:700}
.num{font-variant-numeric:tabular-nums;font-weight:800;letter-spacing:-.02em}
.green{color:var(--green)}.red{color:var(--red)}.dim{color:var(--dim)}

/* hero */
.hero{display:flex;align-items:center;gap:14px;padding:13px 18px;border-left:5px solid var(--line2)}
.hero[data-a=green]{border-left-color:var(--green)}
.hero[data-a=red]{border-left-color:var(--red)}
.hero[data-a=amber]{border-left-color:var(--amber)}
.hero[data-a=slate]{border-left-color:var(--accent)}
.badge{flex:0 0 46px;height:46px;border-radius:13px;display:grid;place-items:center}
.badge svg{width:26px;height:26px}
.b-green{background:rgba(52,211,153,.14)}.b-red{background:rgba(251,113,133,.14)}
.b-amber{background:rgba(251,191,36,.14)}.b-slate{background:rgba(138,162,255,.14)}
.hero .tx{font-size:19px;font-weight:750;letter-spacing:-.01em;line-height:1.2}
.hero .sub{font-size:12.5px;color:var(--dim);margin-top:3px;display:flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--dim)}
.dot.live{background:var(--green);animation:pulse 2s infinite}.dot.bad{background:var(--red)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.5)}70%{box-shadow:0 0 0 7px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}

/* kpi */
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.kpi{padding:12px 14px;display:flex;flex-direction:column;gap:6px;min-height:0}
.kpi .top{display:flex;align-items:center;gap:7px}
.kpi .ic{width:18px;height:18px;color:var(--dim);flex:0 0 18px}
.kpi .v{font-size:24px}
.kpi .s{font-size:11.5px;color:var(--dim);margin-top:auto}
#spark{height:22px;margin-top:auto;width:100%}

/* bet */
.bet{padding:14px 18px;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.bet .head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px}
#betBody{display:flex;flex-direction:column;flex:1;min-height:0}
.pill{padding:5px 13px;border-radius:30px;font-weight:750;font-size:13.5px;display:inline-flex;gap:7px;align-items:center}
.pill.up{background:rgba(52,211,153,.15);color:var(--green)}
.pill.down{background:rgba(251,113,133,.15);color:var(--red)}
.bar{position:relative;height:13px;border-radius:30px;margin:42px 6px 6px;
 background:linear-gradient(90deg,rgba(251,113,133,.85),rgba(251,191,36,.55) 50%,rgba(52,211,153,.85))}
.bar .mk{position:absolute;top:50%;transform:translate(-50%,-50%)}
.bar .entry{width:2px;height:22px;background:var(--dim);border-radius:2px;transform:translate(-50%,-50%);opacity:.7}
.bar .now{width:17px;height:17px;border-radius:50%;background:#fff;border:3px solid #0b1220;box-shadow:0 2px 8px rgba(0,0,0,.6)}
.bar .tag{position:absolute;transform:translateX(-50%);font-size:11px;font-weight:700;color:var(--dim);white-space:nowrap}
.barends{display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin:4px 6px 0}
.plain{background:rgba(0,0,0,.22);border:1px solid var(--line);border-radius:11px;
 padding:11px 13px;font-size:13.5px;margin-top:10px;color:#dbe6f5}
.chartwrap{position:relative;flex:1;min-height:110px;margin:10px 2px 2px}
.chart{width:100%;height:100%;display:block;border-radius:8px;
 background:linear-gradient(180deg,rgba(255,255,255,.02),rgba(0,0,0,.12))}
.lab{position:absolute;right:5px;transform:translateY(-50%);font-size:10.5px;font-weight:700;
 background:rgba(7,11,18,.74);padding:1px 6px;border-radius:6px;pointer-events:none;white-space:nowrap}
.lab.tp{color:var(--green)}.lab.sl{color:var(--red)}.lab.now{color:#fff}

/* panels (right column) */
.panel{display:flex;flex-direction:column;min-height:0;padding:12px 15px}
.panel h2{margin:0 0 8px;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);font-weight:700}
.scroll{overflow:auto;min-height:0;flex:1}
.ev{list-style:none;padding:0;margin:0}
.ev li{position:relative;padding:7px 0 7px 22px;border-bottom:1px solid var(--line);display:flex;gap:10px;font-size:13.5px}
.ev li:last-child{border-bottom:0}
.ev .d{position:absolute;left:2px;top:13px;width:8px;height:8px;border-radius:50%;background:var(--dim)}
.ev .d.open{background:var(--blue)}.ev .d.win{background:var(--green)}.ev .d.loss{background:var(--red)}
.ev .d.veto{background:var(--amber)}.ev .d.ok{background:var(--green)}.ev .d.warn{background:var(--amber)}
.ev .d.dec{background:var(--accent)}.ev .d.think{background:var(--blue)}
.ev .t{color:var(--dim);font-variant-numeric:tabular-nums;flex:0 0 92px;font-size:12px;white-space:nowrap}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th,.tbl td{padding:7px 10px;text-align:left}
.tbl thead th{position:sticky;top:0;background:#101826;color:var(--dim);font-weight:600;font-size:10px;letter-spacing:.07em;text-transform:uppercase}
.tbl tbody tr{border-top:1px solid var(--line)}
.chip{padding:2px 9px;border-radius:7px;font-weight:700;font-size:12px}
.chip.win{background:rgba(52,211,153,.15);color:var(--green)}
.chip.loss{background:rgba(251,113,133,.15);color:var(--red)}
</style></head><body>
<a href="/reasoning" style="position:fixed;top:14px;right:16px;z-index:20;background:#1b2740;border:1px solid #2b3a52;color:#cdd9ee;padding:7px 12px;border-radius:9px;text-decoration:none;font-size:13px;font-weight:700">🧠 Ragionamenti →</a>
<div class="app">

  <div class="hero card" id="hero" data-a="slate">
    <div class="badge b-slate" id="badge"></div>
    <div><div class="tx" id="heroText">…</div>
         <div class="sub"><span class="dot" id="dot"></span><span id="heroStatus"></span></div></div>
  </div>

  <div class="main">
    <div class="col left">
      <div class="kpis">
        <div class="kpi card"><div class="top"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="20" height="13" rx="2"/><path d="M16 12h.01M2 10h20"/></svg><span class="micro">Soldi adesso</span></div>
          <div class="v num" id="equity">—</div><div class="s" id="startLine"></div></div>
        <div class="kpi card"><div class="top"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l6-6 4 4 7-7"/><path d="M14 8h7v7"/></svg><span class="micro">Guadagno / Perdita</span></div>
          <div class="v num" id="pnl">—</div><svg id="spark" viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
        <div class="kpi card"><div class="top"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7h18v4a2 2 0 000 4v2H3v-2a2 2 0 000-4z"/></svg><span class="micro">Operazioni fatte</span></div>
          <div class="v num" id="trades">—</div><div class="s" id="wl"></div></div>
      </div>
      <div class="bet card">
        <div class="head"><span class="micro">Scommessa in corso</span><span id="betPill"></span></div>
        <div id="betBody" style="min-height:0"></div>
      </div>
    </div>

    <div class="col right">
      <div class="panel card"><h2>Ultime mosse</h2><ul class="ev scroll" id="events"></ul></div>
      <div class="panel card"><h2>Operazioni chiuse</h2>
        <div class="scroll"><table class="tbl"><thead><tr><th>Quando</th><th>Tipo</th><th>Esito</th><th>Risultato</th></tr></thead>
        <tbody id="closed"></tbody></table></div></div>
    </div>
  </div>
</div>

<script>
const f=(x,d=2)=>x==null?"—":Number(x).toFixed(d);
const fp=x=>{if(x==null)return"—";x=Number(x);const a=Math.abs(x);
 const d=a>=100?2:a>=1?3:a>=0.01?4:6;return x.toFixed(d);};
const clamp=x=>Math.max(0,Math.min(1,x));
function icon(kind,up){
 const A='stroke="#0b1220" stroke-width="0"';
 if(kind==='open') return up
   ? '<svg viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.4"><path d="M5 17L13 9l3 3 4-5"/><path d="M14 7h6v6"/></svg>'
   : '<svg viewBox="0 0 24 24" fill="none" stroke="#fb7185" stroke-width="2.4"><path d="M5 7l8 8 3-3 4 5"/><path d="M14 17h6v-6"/></svg>';
 if(kind==='think') return '<svg viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';
 if(kind==='bad') return '<svg viewBox="0 0 24 24" fill="none" stroke="#fb7185" stroke-width="2.2"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>';
 return '<svg viewBox="0 0 24 24" fill="none" stroke="#8aa2ff" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>';
}
function spark(pts){if(!pts||pts.length<2)return"";
 const mn=Math.min(...pts),mx=Math.max(...pts),r=(mx-mn)||1,up=pts[pts.length-1]>=pts[0];
 const c=up?'#34d399':'#fb7185';
 const P=pts.map((p,i)=>[(i/(pts.length-1)*300),(40-(p-mn)/r*34)]);
 const line=P.map(p=>p[0].toFixed(1)+","+p[1].toFixed(1)).join(" ");
 const area=`0,44 `+line+` 300,44`;
 return `<polygon points="${area}" fill="${c}" opacity=".10"/><polyline points="${line}" fill="none" stroke="${c}" stroke-width="2.5"/>`;}
function frac(b,price){return clamp(b.up?(price-b.stop)/(b.target-b.stop):(b.stop-price)/(b.stop-b.target));}
function chart(b){
 const pts=(b.path&&b.path.length)?b.path:[b.entry];
 const vals=pts.concat([b.stop,b.target,b.entry]);
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.10||1;lo-=pad;hi+=pad;
 const Y=v=>(100*(1-(v-lo)/(hi-lo)));
 const X=i=>(pts.length<2?0:100*i/(pts.length-1));
 const line=pts.map((p,i)=>X(i).toFixed(2)+","+Y(p).toFixed(2)).join(" ");
 const area=`0,100 `+line+` ${X(pts.length-1).toFixed(2)},100`;
 const hl=(v,c,d)=>`<line x1="0" y1="${Y(v).toFixed(2)}" x2="100" y2="${Y(v).toFixed(2)}" stroke="${c}" stroke-width="1.5" stroke-dasharray="${d}" vector-effect="non-scaling-stroke"/>`;
 const col=b.up?'#34d399':'#fb7185';
 const svg=`<svg class="chart" viewBox="0 0 100 100" preserveAspectRatio="none">
   <polygon points="${area}" fill="${col}" opacity=".07"/>
   ${hl(b.target,'#34d399','4 3')}${hl(b.stop,'#fb7185','4 3')}${hl(b.entry,'#7b8aa0','2 4')}
   <polyline points="${line}" fill="none" stroke="#dbe6f5" stroke-width="2" vector-effect="non-scaling-stroke"/>
  </svg>`;
 const labs=`<div class="lab tp" style="top:${Y(b.target).toFixed(1)}%">🎯 obiettivo ${fp(b.target)}</div>
   <div class="lab sl" style="top:${Y(b.stop).toFixed(1)}%">🛑 stop ${fp(b.stop)}</div>`+
   (b.now!=null?`<div class="lab now" style="top:${Y(b.now).toFixed(1)}%;color:${b.winning?'#34d399':b.winning===false?'#fb7185':'#fff'}">ora ${fp(b.now)}${b.winning==null?'':b.winning?' ✓':' ✕'}</div>`:'');
 return svg+labs;
}

async function tick(){
 let s;try{s=await(await fetch('/api')).json();}catch(e){document.getElementById('heroText').textContent='Dashboard scollegata…';return;}
 const k=s.head.kind, up=s.bet&&s.bet.up;
 const a=k==='open'?(up?'green':'red'):k==='think'?'amber':k==='bad'?'red':'slate';
 const hero=document.getElementById('hero');hero.dataset.a=a;
 const badge=document.getElementById('badge');badge.className='badge b-'+a;badge.innerHTML=icon(k,up);
 document.getElementById('heroText').textContent=s.head.text;
 const dot=document.getElementById('dot');
 dot.className='dot '+(s.stale?'bad':(s.last_log?'live':''));
 document.getElementById('heroStatus').textContent=
   s.last_log==null?'in attesa del primo aggiornamento':
   (s.stale?('fermo da '+Math.round(s.age_min)+' min'):('aggiornato alle '+s.last_log+' · tutto regolare'));

 document.getElementById('equity').textContent=s.equity==null?'—':f(s.equity)+' $';
 document.getElementById('startLine').textContent=s.start==null?'':('partito da '+f(s.start)+' $');
 const p=document.getElementById('pnl');
 p.textContent=(s.pnl>=0?'+':'')+f(s.pnl)+' $  ('+(s.ret>=0?'+':'')+f(s.ret)+'%)';
 p.className='v num '+(s.pnl>=0?'green':'red');
 document.getElementById('spark').innerHTML=spark(s.equity_curve);
 document.getElementById('trades').textContent=s.trades;
 document.getElementById('wl').textContent=s.trades?(s.wins+' vinte · '+s.losses+' perse'):'ancora nessuna';

 const pill=document.getElementById('betPill'), body=document.getElementById('betBody');
 if(s.bet){const b=s.bet;
  pill.innerHTML=`<span class="pill ${b.up?'up':'down'}">${b.dir_h}</span>`;
  body.innerHTML=`<div class="chartwrap">${chart(b)}</div><div class="plain">${b.plain}</div>`;
 }else{pill.innerHTML='';
  body.innerHTML='<div class="dim" style="padding:6px 0 4px">Nessuna scommessa aperta — il bot sta aspettando il momento giusto.</div>';}

 document.getElementById('events').innerHTML=(s.events||[]).slice().reverse()
   .map(e=>`<li><span class="d ${e.k||''}"></span><span class="t">${e.t}</span><span>${e.h}</span></li>`).join('')
   ||'<li class="dim">nessuna mossa ancora</li>';
 document.getElementById('closed').innerHTML=(s.closed||[]).slice().reverse()
   .map(t=>`<tr><td class="dim">${t.at}</td><td>${t.up?'📈 Rialzo':'📉 Ribasso'}</td>
     <td><span class="chip ${t.win?'win':'loss'}">${t.win?'obiettivo':'stop'}</span></td>
     <td class="${t.win?'green':'red'} num">${(t.pnl>=0?'+':'')+f(t.pnl)} $</td></tr>`).join('')
   ||'<tr><td colspan="4" class="dim">ancora nessuna operazione chiusa</td></tr>';
}
tick();setInterval(tick,3000);
</script></body></html>"""


_REASONING_PAGE = r"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ragionamento degli analisti</title>
<style>
:root{--bg:#070b12;--card:#101826;--line:#1f2a3b;--txt:#eaf1fb;--dim:#8696ad;
 --green:#34d399;--red:#fb7185;--flat:#8696ad;--accent:#8aa2ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
 font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:18px 16px 60px}
.wrap{max-width:1000px;margin:0 auto}
.top{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.top h1{font-size:18px;margin:0}
a.back{color:var(--accent);text-decoration:none;font-weight:700;font-size:14px}
.dim{color:var(--dim)}
.acard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:16px 0}
.ahead{display:flex;align-items:center;gap:12px;margin-bottom:6px}
.rb{padding:3px 11px;border-radius:8px;font-weight:800;font-size:13px}
.r-up{background:rgba(52,211,153,.16);color:var(--green)}
.r-down{background:rgba(251,113,133,.16);color:var(--red)}
.r-flat{background:rgba(134,150,173,.16);color:var(--flat)}
.ats{color:var(--dim);font-variant-numeric:tabular-nums;font-size:13px}
details{border-top:1px solid var(--line);padding:5px 0}
details summary{cursor:pointer;font-weight:600;color:#cdd9ee;list-style:none;padding:4px 0}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"\25B8  ";color:var(--accent)}
details[open] summary::before{content:"\25BE  "}
pre{white-space:pre-wrap;word-wrap:break-word;margin:8px 0 4px;padding:10px 12px;
 background:#0b1320;border:1px solid var(--line);border-radius:8px;color:#dbe6f5;
 font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
</style></head><body>
<div class="wrap">
 <div class="top"><a class="back" href="/">&larr; Dashboard</a><h1>&#129504; Ragionamento degli analisti</h1></div>
 <p class="dim" id="meta">caricamento…</p>
 <div id="list"></div>
</div>
<script>
function rcls(r){r=(r||'').toLowerCase();
 if(r==='buy'||r==='overweight')return 'r-up';
 if(r==='sell'||r==='underweight')return 'r-down';return 'r-flat';}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function load(){
 let recs;try{recs=await(await fetch('/api/reasoning')).json();}
 catch(e){document.getElementById('meta').textContent='Dashboard scollegata…';return;}
 const meta=document.getElementById('meta');
 if(!recs.length){meta.textContent='Ancora nessuna analisi registrata in questo run.';
  document.getElementById('list').innerHTML='';return;}
 meta.textContent=recs.length+' analisi · la più recente in alto · si aggiorna ogni 5s';
 document.getElementById('list').innerHTML=recs.map((r,ri)=>`
  <div class="acard">
   <div class="ahead"><span class="rb ${rcls(r.rating)}">${r.rating||'—'}</span>
    <span class="ats">${r.date} ${r.ts}</span></div>
   ${r.sections.map(s=>`<details ${ri===0?'open':''}><summary>${s.label}${s.text?'':' · (vuoto)'}</summary><pre>${esc(s.text)||'(vuoto)'}</pre></details>`).join('')}
  </div>`).join('');
}
load();setInterval(load,5000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/reasoning"):
            self._send(json.dumps(_reasoning_records(), default=str).encode(), "application/json")
        elif self.path.startswith("/api"):
            self._send(json.dumps(_snapshot(), default=str).encode(), "application/json")
        elif self.path.startswith("/reasoning"):
            self._send(_REASONING_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(_PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def log_message(self, *args) -> None:
        pass


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", _PORT), _Handler)
    url = f"http://localhost:{_PORT}"
    print(f"Dashboard live su  ->  {url}   (Ctrl-C per fermare)")
    if os.environ.get("DASHBOARD_OPEN", "1") != "0":
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless / no browser is fine
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstop dashboard")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
