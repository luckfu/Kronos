#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SSH_TARGET=${SSH_TARGET:-oracle4C24G}
AUTH_FILE=${KRONOS_AUTH_FILE:-/etc/nginx/.htpasswd_clawd}

if [[ $# -gt 1 ]]; then
  echo "Usage: SSH_TARGET=oracle4C24G $0 [username]" >&2
  exit 2
fi

username=${1:-}
if [[ -z "$username" ]]; then
  read -r -p "New Kronos username: " username
fi
if [[ ! "$username" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Username may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi

echo "Adding/updating '$username' on $SSH_TARGET. The password will not be displayed."
ssh -t "$SSH_TARGET" "sudo htpasswd '$AUTH_FILE' '$username' && sudo nginx -t && sudo systemctl reload nginx"
echo "Kronos access updated for '$username'."
