# One-command run on any machine with an NVIDIA GPU:
#   docker build -t darkroom .
#   docker run --gpus all -p 8000:8000 -v hf:/root/.cache/huggingface darkroom
# API at http://localhost:8000 (/ws, /health, /tiers); fallback UI at /
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY server server
COPY eval eval
ENV DARKROOM_BASELINE=1
WORKDIR /app/server
EXPOSE 8000
CMD ["uvicorn", "darkroom.server:app", "--host", "0.0.0.0", "--port", "8000"]
