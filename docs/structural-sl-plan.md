# Structural SL — piano di implementazione

Stop loss a doppio binario (soft strutturale + hard catastrofale) guidato da un
contratto strutturato del PM, invece dello stop ATR "al tocco".

**Problema** (individuato da Paolo, confermato sui log): gli analisti esprimono
l'invalidazione per *accettazione* ("reclaim and hold above 1940", "15m close
below X"), l'esecutore chiude *al tocco* — un wick di pochi centesimi uccide
tesi corrette. Censimento su 232 decisioni PM: 34% usa semantica
hold/acceptance, 11% dichiara zone, 5 casi dichiarano già spontaneamente due
livelli ("invalidation above 64000 **and especially above** 64114"), 45%
dichiara un orizzonte temporale, **0% dichiara la finestra di conferma** → va
richiesta per contratto.

**Principio**: l'intento si sposta dalla prosa a un contratto. Il PM dichiara
campi, l'esecutore li onora alla lettera, il prompt gli spiega esattamente come
verranno eseguiti. Niente parsing di prosa, mai.

---

## 1. Contratto: blocco `execution_plan` nella decisione del PM

```json
{
  "invalidation_level": 63966.0,
  "invalidation_semantics": "hold",   // "touch" | "close" | "hold"
  "confirm_bars": 2,                  // solo close/hold; null → default (close=1, hold=2)
  "hard_level": 64114.0,              // null → soft + buffer (v. §4.3)
  "horizon_minutes": 120              // null → time-stop di default
}
```

Regole del contratto:
- **Numeri concreti, non indicatori**: il prompt impone di congelare VWAP/EMA
  al valore corrente ("give concrete price levels"). Le ancore dinamiche si
  materializzano alla fonte.
- **Niente zone nello schema**: chiedendo un singolo numero, la
  disambiguazione bordo-vicino/bordo-lontano avviene nella testa del PM,
  l'unico posto dove può avvenire correttamente.
- Il blocco è richiesto nel prompt **solo quando il flag è attivo** (via
  config, come gli altri fix): aggiungerlo a tutti i profili cambierebbe le
  decisioni anche del gruppo di controllo e sporcherebbe l'A/B.
- `semantics: "touch"` → soft ≡ hard al livello dichiarato (stop singolo, come
  oggi ma sul livello del PM).

Aggiornamento scenario prompt (stesso commit — lezione della sessione): *"il
tuo stop viene eseguito esattamente come lo dichiari: touch = immediato sul
range 1m; close/hold = confermato su chiusure 15m; il tuo hard_level esegue al
tocco e determina size e rischio — dichiaralo fuori dal rumore dei wick"*.

## 2. Normalizzazione basis spot/perp ⚠️ CRITICO

I livelli del PM nascono sui dati **spot** (la pipeline intraday legge
`SOL/USD`, `HYPE/USD` da Kraken spot via ccxt); l'esecuzione avviene sul
**perp** (`PF_SOLUSD`), che quota con un basis. Oggi pochi bps, ma con funding
teso si allarga — e con soft a 0.3-0.5% dal prezzo, 10-15 bps di scarto
sistematico spostano materialmente il trigger.

- Alla decisione: congelare `ratio = last_perp / last_spot`.
- Tutti i livelli del blocco vengono traslati: `level_exec = level × ratio`.
- Il ratio si salva nella posizione (audit + restart).

## 3. Validazione a cascata (prima di fidarsi del blocco)

In ordine; **prima violazione → fallback allo stop ATR attuale** (mai peggio
di oggi) + log `invalidation block rejected: <motivo>` + evento contatore:

1. Blocco presente e campi tipati correttamente
2. `invalidation_level` dal lato giusto rispetto al **prezzo di esecuzione
   attuale** (non solo rispetto all'entry teorico del PM)
3. `hard_level` oltre il soft (lato avverso) se dichiarato
4. Distanza hard entro [0.2×ATR, 5×ATR]
5. `confirm_bars` ∈ [1, 4]

Caso speciale — **già invalidato all'ingresso → SKIP del trade** (non
fallback): tra decisione e apertura passano minuti; se all'apertura il prezzo
ha già superato il soft, la tesi è nata morta. Veto con log
`entry VETOED by invalidation gate: thesis already invalidated at execution`
(coerente con la filosofia del target gate: piuttosto non entrare).

## 4. Esecutore: doppio binario

### 4.1 Trigger

| Livello | Trigger | Ruolo |
|---|---|---|
| Soft | secondo semantica: `touch` = range 1m (come oggi); `close` = chiusura 15m oltre; `hold` = N chiusure 15m **consecutive** oltre | invalidazione della tesi |
| Hard | sempre al tocco sul range 1m, istantaneo | assicurazione catastrofale |

Regole di conferma 15m:
- **La barra a cavallo dell'ingresso non conta**: valgono solo barre 15m che
  *aprono dopo* l'apertura della posizione (la barra in corso era mezza
  scritta, magari proprio sul wick d'ingresso → falsi scatti al primo boundary).
- **Finalità della barra**: si interroga solo la barra con
  `timestamp_open ≤ now − 15m` (mai la candela in formazione).
- **Reset del contatore**: `hold` a N barre = N *consecutive*; una chiusura di
  rientro azzera il conteggio.
- **Persistenza**: contatore di conferma, livelli tradotti, ratio basis e
  semantica vivono nel record posizione in `state.json` — un restart del
  daemon riprende il conteggio da dov'era, non da zero.

### 4.2 Sizing, TP e gate: tutto sulla distanza HARD

Il rischio 2% (`RISK_PCT_PER_TRADE`) si calcola sulla **distanza hard** — lo
scenario peggiore prezzato nella size. L'uscita tipica (soft confermato) costa
meno del budget → la perdita media scende sotto il 2%.

Coerenza geometrica (decisione esplicita, non emergente): **TP = RR × distanza
hard**, e il target gate (`MIN_ANALYST_RR`) confronta il target del PM con la
distanza hard. Conseguenza accettata: hard più largo → TP più lontani → win
rate leggermente giù, vincite più grandi.

### 4.3 Buffer di default (hard assente)

`hard = soft + 0.4 × ATR × confirm_bars` (lato avverso). Il buffer **scala con
le barre di conferma**: l'escursione avversa attesa nella finestra "hold" a 2
barre è maggiore di quella a 1. Costante 0.4 da tarare coi dati.

### 4.4 Precedenza nella stessa candela 1m

Pessimistica, coerente con `check_hit` attuale: hard prima di TP. Il soft non
compete nella stessa candela (scatta solo a chiusura 15m, valutata al tick
successivo al boundary).

### 4.5 Interazione con gli altri fix

- **Regime-exit**: resta attivo (una sola variabile cambia vs profilo tuned),
  ma può chiudere prima che il soft si pronunci, mascherando wick-saves →
  ogni uscita etichettata col proprio meccanismo (§5), analisi separata.
- **Time-stop**: se `horizon_minutes` dichiarato → time-stop = 2×orizzonte;
  altrimenti fallback `TIME_STOP_HOURS` (6h). Copre in modo grezzo anche la
  stagnazione sul livello (v. fuori scope).

## 5. Logging, eventi, metriche

Nuove etichette d'uscita: `SOFT` (invalidazione confermata) distinta da `SL`
(hard al tocco). Dashboard: testi dedicati ("Chiusa: invalidazione confermata
(15m)" vs "Chiusa: stop di protezione").

Eventi/metriche per il verdetto:
1. **Wick-save**: range 1m oltre il soft senza conferma successiva → evento
   `soft_touched_not_confirmed`; a fine trade si registra l'esito. Conta i
   trade che il vecchio stop avrebbe ucciso e il soft ha salvato.
2. **Costo del ritardo soft**: per ogni uscita SOFT, logga
   `exec_price − soft_level` (si esce dopo la chiusura oltre il livello, con
   fino a ~60s di tick: slittamento sistematico). Verdetto = wick-saves
   guadagnati − questo premio assicurativo. Senza questa colonna l'A/B mente.
3. **Tasso di rigetto blocchi**: contatore per motivo (§3). Se il PM sbaglia
   troppo spesso, si raffina il contratto.
4. **Perdita media vs budget**: deve stare sotto il 2% (uscite soft < hard).

## 6. Config e rollout

- Flag: `STRUCTURAL_SL=1` (env, default off — stessa architettura opt-in
  degli altri fix; a flag spento tutto byte-identico a oggi).
- Profilo `struct` = profilo tuned + `STRUCTURAL_SL` — misura il valore
  marginale della sola feature; le tuned diventano gruppo di controllo.
- **Quattro simboli, non due**: a ~3-4 trade/giorno/simbolo e wick-saves su
  una minoranza di trade, con 2 simboli servono 2-3 settimane di dati; con
  SOL+HYPE+ADA+BTC struct il verdetto arriva in ~1 settimana. La VPS regge
  (misurato: ~300MB/daemon, 1.8GB liberi con 2 attivi).
- Porte 8771-8774, host Caddy `solstruct|hypestruct|adastruct|btcstruct.…`
  (ricordare: dopo il pull, `--force-recreate caddy` — il bind-mount del
  Caddyfile resta sul vecchio inode).

Criteri di decisione dopo ~1 settimana: wick-saves netti > 0 (metrica 1 −
metrica 2), perdita media < budget, rigetti < ~20%. Se il PM produce blocchi
validi raramente → prima raffinare il contratto, poi rivalutare.

## 7. Fuori scope v1 (deliberato)

- **Stagnazione sul livello** (prezzo pinned sul soft senza confermare):
  coperta grossolanamente dal time-stop a orizzonte; versione fine dopo, coi
  dati.
- **Lato ingressi** (stesso gap semantico sui breakout): il controfattuale del
  18/07 ha mostrato valore minore del lato uscite.
- **Parsing della prosa** come ripiego: mai. O blocco valido o ATR.

## 8. File toccati (checklist)

| File | Modifica |
|---|---|
| `tradingagents/graph/…` (signal/PM) | richiesta blocco `execution_plan` (structured output, solo con flag) + estrazione |
| `live/analysis.py` | flag in config; passaggio `execution_plan` + ratio basis a valle |
| `tradingagents/agents/utils/agent_utils.py` | scenario prompt condizionale (esecuzione stop dichiarata) |
| `paper_sim/simulator.py` | validazione a cascata; invalidation gate (skip); doppio binario in `monitor()`; fetch/allineamento barre 15m; sizing/TP/target-gate su hard; persistenza contatori in posizione; eventi e metriche |
| `paper_sim/run.py` | env `STRUCTURAL_SL` |
| `paper_sim/dashboard.py` | testi per `SOFT`, invalidation gate, wick-save nel feed |
| `docker-compose.paper.yml` | 4 servizi struct + dashboard (porte 8771-8774) |
| `deploy/Caddyfile` | 4 host nuovi |

## 9. Test plan (unit, con mock — pattern della sessione)

1. Blocco valido close/1 barra: barra a cavallo ignorata; conferma alla prima
   barra piena oltre → uscita SOFT
2. `hold`/2 barre: oltre→rientro→oltre→oltre = scatta alla seconda consecutiva
   (reset verificato)
3. Wick oltre il soft senza conferma → nessuna uscita + evento wick-save
4. Hard toccato durante la finestra di conferma → uscita SL immediata
5. Già invalidato all'esecuzione → SKIP con veto dedicato
6. Blocco invalido (lato sbagliato / distanza fuori range / bars fuori range)
   → fallback ATR + rejected event
7. Basis: livelli traslati con ratio ≠ 1; ratio persistito
8. Restart a metà conferma: contatore ripreso da `state.json`
9. Sizing e TP calcolati sulla distanza hard; target gate su hard
10. `semantics: touch` → comportamento stop singolo al livello PM
11. Flag spento → byte-identico a oggi (nessun campo richiesto nel prompt)
