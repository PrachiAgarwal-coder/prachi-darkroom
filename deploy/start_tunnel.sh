#!/usr/bin/env bash
# Start the model server on any Linux GPU notebook (Colab, Kaggle, a lab machine) and
# expose it through a free Cloudflare quick tunnel.
#
#   bash deploy/start_tunnel.sh https://your-app.lovable.app
#
# Prints the public server URL and a ready-to-share link for judges.
# Background processes keep running after the notebook cell finishes; rerun to restart.
set -u
APP_URL="${1:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/server"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "!! No GPU on this machine. Enable a GPU (Colab: Runtime > Change runtime type > T4 GPU;"
  echo "!! Kaggle: Settings > Accelerator > GPU T4 x2) and rerun."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Installing dependencies (about 2-3 min)..."
pip install -q -r requirements.txt
pip uninstall -y -q torchao >/dev/null 2>&1 || true   # old torchao makes peft refuse LoRAs

pkill -f "uvicorn darkroom.server" >/dev/null 2>&1 || true
pkill -f "cloudflared tunnel" >/dev/null 2>&1 || true
sleep 2

nohup python -m uvicorn darkroom.server:app --host 0.0.0.0 --port 8000 > "$ROOT/server.log" 2>&1 &
if [ ! -x "$ROOT/cloudflared" ]; then
  curl -sL -o "$ROOT/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$ROOT/cloudflared"
fi
nohup "$ROOT/cloudflared" tunnel --url http://localhost:8000 > "$ROOT/tunnel.log" 2>&1 &

URL=""
for _ in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/tunnel.log" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  sleep 2
done
[ -z "$URL" ] && { echo "!! Tunnel did not start; see $ROOT/tunnel.log"; exit 1; }

printf "Loading models (first time 5-8 min)"
for _ in $(seq 1 240); do
  if curl -s localhost:8000/health | grep -q '"ok":true'; then echo " ready."; break; fi
  if ! pgrep -f "uvicorn darkroom.server" >/dev/null; then echo; echo "!! Server crashed:"; tail -30 "$ROOT/server.log"; exit 1; fi
  printf "."; sleep 5
done

WS="${URL/https:/wss:}/ws"
echo
echo "Model server (has a simple built-in UI): $URL"
echo "WebSocket URL:                           $WS"
[ -n "$APP_URL" ] && echo ">>> LINK FOR JUDGES: ${APP_URL%/}/?server=$WS"
echo "Logs: tail -f $ROOT/server.log"
