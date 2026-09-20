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
cp "$PROJECT_DIR/webui/auth.py" "$release_dir/webui/auth.py"
cp "$PROJECT_DIR/webui/hermes_analysis.py" "$release_dir/webui/hermes_analysis.py"
cp "$PROJECT_DIR/webui/templates/index.html" "$release_dir/webui/templates/index.html"
cp "$PROJECT_DIR/webui/templates/daily_rankings.html" "$release_dir/webui/templates/daily_rankings.html"
cp "$PROJECT_DIR/webui/templates/login.html" "$release_dir/webui/templates/login.html"
cp "$PROJECT_DIR/webui/size_reference.json" "$release_dir/webui/size_reference.json"
cp "$PROJECT_DIR/webui/sector_vocabulary.json" "$release_dir/webui/sector_vocabulary.json"
cp "$PROJECT_DIR/webui/symbol_sector_map.json" "$release_dir/webui/symbol_sector_map.json"
cp "$PROJECT_DIR/webui/update_sector_mapping.py" "$release_dir/webui/update_sector_mapping.py"
cp "$SCRIPT_DIR/requirements.txt" "$release_dir/deploy/requirements.txt"
cp "$SCRIPT_DIR/prediction-requirements.txt" "$release_dir/deploy/prediction-requirements.txt"
cp "$SCRIPT_DIR/prediction_drill.py" "$release_dir/deploy/prediction_drill.py"
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
sudo install -o opc -g opc -m 0644 "$stage/webui/auth.py" "$root/webui/auth.py"
sudo install -o opc -g opc -m 0644 "$stage/webui/hermes_analysis.py" "$root/webui/hermes_analysis.py"
sudo install -o opc -g opc -m 0644 "$stage/webui/templates/index.html" "$root/webui/templates/index.html"
sudo install -o opc -g opc -m 0644 "$stage/webui/templates/daily_rankings.html" "$root/webui/templates/daily_rankings.html"
sudo install -o opc -g opc -m 0644 "$stage/webui/templates/login.html" "$root/webui/templates/login.html"
sudo install -o opc -g opc -m 0644 "$stage/webui/size_reference.json" "$root/webui/size_reference.json"
sudo install -o opc -g opc -m 0644 "$stage/webui/sector_vocabulary.json" "$root/webui/sector_vocabulary.json"
sudo install -o opc -g opc -m 0755 "$stage/webui/update_sector_mapping.py" "$root/webui/update_sector_mapping.py"
if [[ ! -f "$root/webui/symbol_sector_map.json" ]]; then
  sudo install -o opc -g opc -m 0644 "$stage/webui/symbol_sector_map.json" "$root/webui/symbol_sector_map.json"
fi
if [[ -f "$root/webui/sector_reference.json" ]]; then
  sudo mv "$root/webui/sector_reference.json" "$root/data/sector_reference.legacy.$stamp.json"
fi
sudo install -o opc -g opc -m 0644 "$stage/deploy/requirements.txt" "$root/deploy/requirements.txt"
sudo install -o opc -g opc -m 0644 "$stage/deploy/prediction-requirements.txt" "$root/deploy/prediction-requirements.txt"
sudo install -o opc -g opc -m 0755 "$stage/deploy/prediction_drill.py" "$root/deploy/prediction_drill.py"
if [[ ! -x "$root/.venv/bin/python" ]]; then python3 -m venv "$root/.venv"; fi
"$root/.venv/bin/python" -m pip install --no-cache-dir --upgrade pip
"$root/.venv/bin/python" -m pip install --no-cache-dir -r "$root/deploy/requirements.txt"
env_file=/opt/kronos-web/data/kronos-web.env
if [[ ! -f "$env_file" ]] || ! sudo grep -q '^KRONOS_SECRET_KEY=' "$env_file"; then
  secret=$("$root/.venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))')
  tmp_env=$(mktemp)
  if [[ -f "$env_file" ]]; then sudo cat "$env_file" > "$tmp_env"; fi
  printf 'KRONOS_SECRET_KEY=%s\n' "$secret" >> "$tmp_env"
  sudo install -o opc -g opc -m 0600 "$tmp_env" "$env_file"
  rm -f "$tmp_env"
  unset secret
fi
htpasswd=/etc/nginx/.htpasswd_clawd
if [[ -f "$htpasswd" ]] && [[ ! -r "$htpasswd" ]]; then
  sudo setfacl -m u:opc:r "$htpasswd" || sudo chmod a+r "$htpasswd" || true
fi
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
