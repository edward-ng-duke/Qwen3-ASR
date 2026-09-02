#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ss -H -ltn '( sport = :6002 )' | grep -q .; then
  echo 'host port 6002 is already occupied; refusing to replace an unknown service' >&2
  exit 1
fi

docker compose up -d --no-build
docker compose ps
