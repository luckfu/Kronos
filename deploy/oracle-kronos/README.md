# Oracle Kronos Small 0.1 Web Gateway

This directory deploys only the lightweight Kronos web UI and market-data gateway. The Oracle host does not receive a model or training dependencies. Inference is sent to the Modal service backed by `luckfu/Kronos-small-0.1-Cosine-C2-Best` `Segment@179`.

## Architecture

```text
Browser -> https://allmoneybymehold.com/kronos/
        -> Nginx reverse proxy (no Basic Auth)
        -> Gunicorn at 127.0.0.1:7072
        -> Flask cookie session, checked against /etc/nginx/.htpasswd_clawd
        -> incremental market-data cache and collection
        -> fixed sector vocabulary + replaceable symbol mapping
        -> full-market size-percentile reference
        -> Small 0.1 Modal Serverless inference API
```

The deployment creates only `/opt/kronos-web`, an isolated `kronos-web.service`, and `/etc/nginx/default.d/kronos.conf`. It does not modify the main Nginx file or stop existing services. Existing Kronos-specific files are timestamp-backed up before replacement. Supabase is not used. Prediction records are stored as JSON under `/opt/kronos-web/data/prediction_results`, survive redeployments, and are grouped in the UI by their market-data cutoff date rather than submission mode. Adjusted daily market data is cached per stock under `/opt/kronos-web/data/market_data_cache`; refreshes request a small overlap after the cached last date, merge and deduplicate rows, and then send only the requested context to Modal.

## Deploy

Prerequisites are the SSH alias `oracle4C24G`, `ssh`, and `rsync`. The server needs Python 3, Nginx, and systemd.

```bash
bash deploy/oracle-kronos/deploy.sh
```

A different alias can be supplied with `SSH_TARGET=opc@example-host`. The service enforces `KRONOS_REMOTE_ONLY=1`, so request payloads cannot select local inference. Deploy the Modal Small App first; the existing URL remains `https://luckfu--kronos-beta-v1-2-inference-web.modal.run`.

`/kronos/` is no longer protected by Nginx Basic Auth. The Flask app shows a
mobile-friendly login form and stores an HttpOnly, Secure, SameSite=Lax cookie
(`kronos_session`, path `/kronos`). Credentials are still validated against the
existing `/etc/nginx/.htpasswd_clawd` file (apr1 or bcrypt, as `htpasswd`
writes). The deployment never creates, replaces, or prints that password file;
it only needs the `opc` service user to be able to read it. Remember-me
defaults to on and lasts 30 days (`KRONOS_SESSION_DAYS`).

The cookie is signed with `KRONOS_SECRET_KEY`. Generate it once on Oracle and
keep it in `/opt/kronos-web/data/kronos-web.env` (mode `0600`), which the
systemd unit loads via `EnvironmentFile`. The deploy script creates this file
on first install if the key is missing:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
# store as KRONOS_SECRET_KEY=... in /opt/kronos-web/data/kronos-web.env
```

Do not commit or rotate that value casually: changing it signs every existing
session out. The first deploy after this change must reload Nginx (to drop
`auth_basic`) and restart `kronos-web` so the new login code and environment
file are picked up. `deploy.sh` already does both.

## Update Industry Mapping

The Beta V1.2 sector vocabulary and ID order are immutable. The per-symbol mapping can be
refreshed manually from BaoStock and is preserved by later web deployments:

```bash
bash deploy/oracle-kronos/update-sector-mapping.sh
```

An optional historical snapshot date can be supplied as `YYYY-MM-DD`. The updater validates
the fixed vocabulary, refuses unexpectedly small provider responses, creates
`symbol_sector_map.json.bak`, and atomically replaces the mapping. The wrapper restarts the
gateway so both Gunicorn workers use the new mapping.

Add or change a user without exposing the password in shell history:

```bash
bash deploy/oracle-kronos/add-user.sh analyst
```

The script prompts securely on the Oracle host and updates the shared htpasswd
file. Kronos does not need an Nginx reload after a password change; the next
login reads the file. Set `SSH_TARGET=opc@example-host` when using a different
SSH target. Existing users remain valid; use the same command with an existing
username to change its password. Opening `https://allmoneybymehold.com/kronos/`
shows the in-app login form until a session cookie is set. Use **退出** on the
rankings page to clear the cookie.

## Hermes / DeepSeek ranking analysis

The daily-rankings detail card has an on-demand **Hermes 分析** button. Opening
a stock does not call the model. Clicking the button sends `POST
/api/daily-rankings/<asof>/<symbol>/hermes-analysis`. Kronos builds a short
Chinese prompt from the published ranking row (code, name, asof, close, D1,
D10, rank, sector) and execs the local Hermes CLI as `opc`:

```bash
hermes chat -q "PROMPT" --oneshot -Q -m deepseek-v4-pro --provider deepseek
```

It captures stdout only and returns `{analysis, model, provider, elapsed_sec}`.
Kronos never scrapes the Hermes WebUI (`127.0.0.1:18787` / `/hermes/`) and never
reads `~/.hermes/.env`. DeepSeek keys stay with Hermes.

After deploy, `kronos-web.service` must be installed and restarted so the new
route, template, and environment are picked up. `hermes` must remain executable
for `opc` (default `KRONOS_HERMES_BIN=/data/miniconda3/bin/hermes`, with
`/data/miniconda3/bin` on `PATH`). Confirm with:

```bash
ssh oracle4C24G 'sudo -u opc -H /data/miniconda3/bin/hermes -z "ping"'
```

Configurable through the unit file or `/opt/kronos-web/data/kronos-web.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `KRONOS_HERMES_BIN` | `/data/miniconda3/bin/hermes` | CLI path |
| `KRONOS_HERMES_PROVIDER` | `deepseek` | Forced provider (not the Hermes default gpt-5.5) |
| `KRONOS_HERMES_MODEL` | `deepseek-v4-pro` | Forced model |
| `KRONOS_HERMES_TIMEOUT` | `120` | Subprocess timeout in seconds (capped at 180) |
| `KRONOS_HERMES_CACHE_TTL` | `1800` | In-process cache for repeat clicks; `0` disables |
| `KRONOS_HERMES_DISABLED` | unset | Set `1` to return HTTP 503 without calling Hermes |

If the binary is missing or the flag is set, the API returns HTTP 503 with an
`error` string the detail card displays. Timeouts return HTTP 504. Live Oracle
smoke: search `000063` / `sz.000063`, open the card, click **Hermes 分析**, and
wait for DeepSeek text under the history+forecast chart. CI uses a mocked
subprocess and does not call DeepSeek.

## Verify

This checks health, model status, and performs one billable Modal prediction for `600519`:

```bash
bash deploy/oracle-kronos/test.sh
```

Useful diagnostics:

```bash
ssh oracle4C24G 'sudo systemctl status kronos-web --no-pager'
ssh oracle4C24G 'sudo journalctl -u kronos-web -n 100 --no-pager'
ssh oracle4C24G 'curl -fsS http://127.0.0.1:7072/health'
```

## Full-market daily prediction

`prediction_drill.py` is a portable, read-only data-to-inference runner. It can
run from a developer machine or Oracle, connects directly to Supabase through
`DB_URL` or `DATABASE_URL`, builds the strict full-market universe, and calls
Modal in batches of at most 12 symbols. Completed batch files are reusable, so
an interrupted run resumes without paying for successful batches again.

Install the standalone runner dependencies into the current Conda environment:

```bash
python -m pip install -r deploy/oracle-kronos/prediction-requirements.txt
```

Run locally:

```bash
export DATABASE_URL='postgresql://...'
python deploy/oracle-kronos/prediction_drill.py \
  --asof 2026-09-18 \
  --sector-map webui/symbol_sector_map.json \
  --output-dir data/prediction_shadow/2026-09-18/full_market
```

Run on Oracle with the same program and its existing environment file:

```bash
set -a
. /home/opc/.openclaw/.env
set +a
python /opt/kronos-web/deploy/prediction_drill.py \
  --asof 2026-09-18 \
  --sector-map /opt/kronos-web/webui/symbol_sector_map.json \
  --output-dir /opt/kronos-web/data/prediction_shadow/2026-09-18/full_market
```

The default is the complete eligible universe, 12 symbols per request and 5
samples per symbol. `--limit 12` is available for a billable smoke test. Future
timestamps come from the A-share exchange calendar exposed by AkShare; a run
fails rather than silently substituting weekdays when the calendar is absent.
Artifacts include the data audit, frozen universe, one atomic JSON file per
batch, progress state, complete JSONL predictions and the D10 ranking CSV. The
runner never writes to Supabase or the web UI's production prediction folder.

## Release Contents

Only the Web gateway, template, size reference, fixed sector vocabulary, initial symbol-sector
mapping, mapping updater, and deployment configuration files are uploaded. Model directories,
checkpoints, datasets, outputs, artifacts, training code, and credentials are never included.
After the first installation, ordinary deployments preserve the Oracle host's current
`symbol_sector_map.json`; only the manual updater replaces it.

The Oracle gateway stores per-stock adjusted daily cache files in `/opt/kronos-web/data/market_data_cache`, a durable A-share name table in `/opt/kronos-web/data/stock_name_cache.json`, and prediction snapshots in `/opt/kronos-web/data/prediction_results`. Those directories are persistent runtime data and are not replaced by the release upload. Gunicorn workers preload the name table from disk at startup and refresh BaoStock in the background when the file is missing or older than a day, so the daily rankings page does not wait on `query_stock_basic` after a restart. The gateway owns market-data collection and derives the signal-date industry ID and continuous size percentile. Modal receives only the prepared 120-row OHLCVA context, 10 future timestamps, and those two conditions.
