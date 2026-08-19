#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
SSH_TARGET=${SSH_TARGET:-oracle4C24G}
REMOTE_STAGE=/tmp/kronos-web-release

release_dir=$(mktemp -d)
trap 'rm -rf "$release_dir"' EXIT

mkdir -p "$release_dir/webui/templates" "$release_dir/deploy"
cp "$PROJECT_DIR/webui/app.py" "$release_dir/webui/app.py"
cp "$PROJECT_DIR/webui/templates/index.html" "$release_dir/webui/templates/index.html"
cp "$PROJECT_DIR/webui/size_reference.json" "$release_dir/webui/size_reference.json"
cp "$SCRIPT_DIR/requirements.txt" "$release_dir/deploy/requirements.txt"
cp "$SCRIPT_DIR/kronos-web.service" "$release_dir/deploy/kronos-web.service"
cp "$SCRIPT_DIR/nginx-kronos-location.conf" "$release_dir/deploy/nginx-kronos-location.conf"

echo "Uploading lightweight Kronos gateway to $SSH_TARGET..."
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_STAGE'"
rsync -az --delete "$release_dir/" "$SSH_TARGET:$REMOTE_STAGE/"

ssh "$SSH_TARGET" 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail
stage=/tmp/kronos-web-release
root=/opt/kronos-web
stamp=$(date +%Y%m%d%H%M%S)
sudo mkdir -p "$root/webui/templates" "$root/deploy" "$root/data/prediction_results" "$root/data/market_data_cache"
sudo chown opc:opc "$root"
sudo chown -R opc:opc "$root/data"
sudo install -o opc -g opc -m 0644 "$stage/webui/app.py" "$root/webui/app.py"
sudo install -o opc -g opc -m 0644 "$stage/webui/templates/index.html" "$root/webui/templates/index.html"
sudo install -o opc -g opc -m 0644 "$stage/webui/size_reference.json" "$root/webui/size_reference.json"
sudo install -o opc -g opc -m 0644 "$stage/deploy/requirements.txt" "$root/deploy/requirements.txt"
if [[ ! -x "$root/.venv/bin/python" ]]; then python3 -m venv "$root/.venv"; fi
"$root/.venv/bin/python" -m pip install --no-cache-dir --upgrade pip
"$root/.venv/bin/python" -m pip install --no-cache-dir -r "$root/deploy/requirements.txt"
if [[ -f /etc/systemd/system/kronos-web.service ]]; then
  sudo cp -a /etc/systemd/system/kronos-web.service "/etc/systemd/system/kronos-web.service.bak.$stamp"
fi
sudo install -m 0644 "$stage/deploy/kronos-web.service" /etc/systemd/system/kronos-web.service
nginx_config=/etc/nginx/default.d/kronos.conf
nginx_backup=
if [[ -f "$nginx_config" ]]; then
  nginx_backup="$nginx_config.bak.$stamp"
  sudo cp -a "$nginx_config" "$nginx_backup"
fi
sudo install -m 0644 "$stage/deploy/nginx-kronos-location.conf" "$nginx_config"
if ! sudo nginx -t; then
  if [[ -n "$nginx_backup" ]]; then sudo cp -a "$nginx_backup" "$nginx_config"; else sudo rm -f "$nginx_config"; fi
  sudo nginx -t || true
  echo "Nginx validation failed; previous Kronos location was restored." >&2
  exit 1
fi
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-web
sudo systemctl restart kronos-web
sudo systemctl reload nginx
for attempt in 1 2 3 4 5; do
  if curl -fsS http://127.0.0.1:7072/health; then printf '\n'; exit 0; fi
  sleep 2
done
sudo journalctl -u kronos-web -n 80 --no-pager
exit 1
REMOTE_SCRIPT

echo "Deployment complete: https://allmoneybymehold.com/kronos/"
