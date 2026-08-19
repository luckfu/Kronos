# Oracle Kronos Web Gateway

This directory deploys only the lightweight Kronos web UI and market-data gateway. The Oracle host does not receive a model, checkpoint, training dataset, PyTorch, ModelScope, or Hugging Face tooling. Inference is sent to the Modal production service using model `luckfu/Kronos-A-Share-Forecast`.

## Architecture

```text
Browser -> https://allmoneybymehold.com/kronos/
        -> existing Nginx Basic Auth (/etc/nginx/.htpasswd_clawd)
        -> Gunicorn at 127.0.0.1:7072
        -> incremental market-data cache and collection
        -> Modal Serverless inference API
```

The deployment creates only `/opt/kronos-web`, an isolated `kronos-web.service`, and `/etc/nginx/default.d/kronos.conf`. It does not modify the main Nginx file or stop existing services. Existing Kronos-specific files are timestamp-backed up before replacement. Supabase is not used. Prediction records are stored as JSON under `/opt/kronos-web/data/prediction_results` and survive redeployments. Adjusted daily market data is cached per stock under `/opt/kronos-web/data/market_data_cache`; refreshes request a small overlap after the cached last date, merge and deduplicate rows, and then send only the requested context to Modal.

## Deploy

Prerequisites are the SSH alias `oracle4C24G`, `ssh`, and `rsync`. The server needs Python 3, Nginx, and systemd.

```bash
bash deploy/oracle-kronos/deploy.sh
```

A different alias can be supplied with `SSH_TARGET=opc@example-host`. The service enforces `KRONOS_REMOTE_ONLY=1`, so request payloads cannot select local inference.

The `/kronos/` location reuses the server's existing `/etc/nginx/.htpasswd_clawd` credentials. The deployment never creates, replaces, or prints that password file.

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

## Release Contents

Only `webui/app.py`, `webui/templates/index.html`, `webui/size_reference.json`, and the three deployment configuration files are uploaded. Model directories, checkpoints, datasets, outputs, artifacts, training code, and credentials are never included.

The Oracle gateway stores per-stock adjusted daily cache files in `/opt/kronos-web/data/market_data_cache` and prediction snapshots in `/opt/kronos-web/data/prediction_results`. Both directories are persistent runtime data and are not replaced by the release upload. The gateway owns market-data collection; Modal receives only the prepared OHLCVA context and performs inference.
