#!/usr/bin/env bash
set -euo pipefail

python tools/build_packet.py --date 2026-01-28 --asof 09:25 --out out/packet.json
python - <<'PY'
import json
from pathlib import Path

path = Path("out/packet.json")
if not path.exists():
    raise SystemExit("packet.json not created")
try:
    json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON: {exc}")
PY
