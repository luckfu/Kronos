#!/usr/bin/env bash
set -euo pipefail
BASE_URL=${KRONOS_PUBLIC_URL:-https://allmoneybymehold.com/kronos}
echo "Health:"
curl --fail --silent --show-error "$BASE_URL/health"
printf '\n'

if [[ -z "${KRONOS_TEST_USER:-}" || -z "${KRONOS_TEST_PASSWORD:-}" ]]; then
  echo "Skipping authenticated API checks. Set KRONOS_TEST_USER and KRONOS_TEST_PASSWORD to exercise model-status and predict."
  exit 0
fi

cookie_jar=$(mktemp)
trap 'rm -f "$cookie_jar"' EXIT
login_body=$(mktemp)
trap 'rm -f "$cookie_jar" "$login_body"' EXIT
python3 - "$login_body" <<'PY'
import os, sys, urllib.parse
path = sys.argv[1]
with open(path, 'w', encoding='utf-8') as handle:
    handle.write(urllib.parse.urlencode({
        'username': os.environ['KRONOS_TEST_USER'],
        'password': os.environ['KRONOS_TEST_PASSWORD'],
        'remember': '1',
    }))
PY

curl --fail --silent --show-error \
  -c "$cookie_jar" -b "$cookie_jar" \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-binary @"$login_body" \
  "$BASE_URL/login" >/dev/null
rm -f "$login_body"

printf '\nModel status:\n'
curl --fail --silent --show-error -b "$cookie_jar" "$BASE_URL/api/model-status"
printf '\n\nSingle-stock Modal prediction:\n'
curl --fail --silent --show-error \
  -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","backend":"remote","sample_count":5,"temperature":0.65,"top_p":0.8}' \
  "$BASE_URL/api/predict"
printf '\n'
