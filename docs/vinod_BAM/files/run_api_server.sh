#!/usr/bin/env bash
set -euo pipefail

python3 -m uvicorn api_server_launcher:create_app --factory --host 0.0.0.0 --port 8000
