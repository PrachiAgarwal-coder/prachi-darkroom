# Backup deployment: free GPU on Google Colab or Kaggle + Cloudflare tunnel

Use this if Modal is not available. The URL changes every time you restart, and the
notebook must stay open during judging, so treat it as plan B.

Runtime → Change runtime type → T4 GPU. Then run these cells:

```bash
!git clone https://github.com/<you>/<repo>.git darkroom
%cd darkroom/server
!pip install -q -r requirements.txt
```

```bash
# start the server in the background (first start downloads ~5 GB of weights, 3-6 min)
!nohup uvicorn darkroom.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
!sleep 30 && tail -5 server.log
```

```bash
# public HTTPS URL (look for https://....trycloudflare.com in the output)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &
!sleep 8 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log
```

Wait until `tail server.log` shows `Application startup complete`, then open the
trycloudflare URL. On a T4 set `DARKROOM_BASELINE=0` if you run out of memory
(`!DARKROOM_BASELINE=0 nohup uvicorn ...`); the race then reports that the
baseline is not loaded.
