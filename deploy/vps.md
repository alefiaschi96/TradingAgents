# Deploy paper-sim su VPS

Obiettivo: run parallele 24/7 (un daemon per simbolo) senza tenere acceso il Mac.
Testato pensando a un VPS da ~4€/mese (Hetzner CX22/CAX11, 4GB RAM regge 3-4 simboli).

## 1. Server

```bash
# sul VPS (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
```

## 2. Codice + segreti

```bash
git clone git@github.com:alefiaschi96/TradingAgents.git && cd TradingAgents
# .env.live NON è nel repo: copialo dal Mac
#   (dal Mac) scp .env.live <vps>:TradingAgents/
```

Servono solo le chiavi LLM: i prezzi Kraken sono pubblici, nessuna chiave di
trading finisce sul server. Nota: da un VPS EU fuori dall'Italia Polymarket
torna raggiungibile — puoi valutare `PREDICTION_MARKETS=true`.

## 3. Avvio

```bash
docker compose -f docker-compose.paper.yml up -d --build
docker compose -f docker-compose.paper.yml logs -f sol   # segui un daemon
```

`restart: unless-stopped` + stato su volume: i daemon sopravvivono a riavvii
del server e riprendono da dove erano (design del daemon: state.json per slug).

## 4. Vedere le dashboard dal Mac

Le porte sono pubblicate solo su 127.0.0.1 del VPS, mai su internet.

**Opzione A — tunnel SSH (zero setup):**
```bash
ssh -N -L 8765:localhost:8765 -L 8766:localhost:8766 <vps>
# poi apri http://localhost:8765 (SOL) e :8766 (ADA)
```

**Opzione B — Tailscale (comodo, sempre attivo):** installa tailscale su VPS e
Mac, poi pubblica le porte sull'IP tailscale del VPS cambiando nel compose
`127.0.0.1:8765:8765` con `<ip-tailscale>:8765:8765`, oppure usa
`tailscale serve`. Le dashboard restano raggiungibili solo dalla tua tailnet.

## 5. Aggiungere un simbolo

Nel compose copia la coppia `sol` / `sol-dashboard`, cambia `SYMBOL`, i due
percorsi `/data/<SLUG>/...` e la porta host della dashboard (8767, ...), poi
`up -d` di nuovo.

## 6. Recuperare stato e log

Tutto vive nel volume `paper_data`:

```bash
docker compose -f docker-compose.paper.yml exec sol ls /data/SOL/logs
# o per copiare in locale:
docker cp $(docker compose -f docker-compose.paper.yml ps -q sol):/data/SOL ./SOL-backup
```

I log NON vengono scritti nel repo del server (PAPER_LOG_DIR punta al volume),
quindi `git pull` per aggiornare il codice non trova mai conflitti.

## Aggiornare il codice

```bash
git pull && docker compose -f docker-compose.paper.yml up -d --build
```
