# System architecture

GitHub renders the diagram below. Every box names the file that implements it.

```mermaid
flowchart LR
    subgraph Client["Browser (phone or laptop) — src/routes/index.tsx (React UI)"]
        UI[Prompt + device/tier picker]
        AD[Adaptive tier selection<br/>navigator.deviceMemory → memory allowance]
        VIEW[Live denoising view<br/>verifier overlay · race · Pareto]
    end

    subgraph Server["GPU server — server/darkroom/server.py (FastAPI)"]
        WS["/ws WebSocket<br/>validation · GPU lock · OOM fallback"]
        TI["/tiers · /health"]
        subgraph Engine["engine.py"]
            ENC[CLIP text encoder<br/>prompt encoding]
            subgraph Tiers["Tiers (config.py)"]
                B[Baseline<br/>SD 1.5 FP32 · 50 steps · CFG 7.5]
                Q[Quality<br/>LCM-LoRA fused · FP16 · 8 steps]
                BA[Balanced<br/>LCM-LoRA · FP16 · 4 steps · TAESD]
                L[Lite<br/>LCM-LoRA · INT8 UNet · 3 steps · 384 px · TAESD]
            end
            DEC[Latent decode<br/>full VAE or TAESD]
            PRV[TAESD step previews]
        end
        POST["postprocess.py<br/>Lanczos upscale · gray-world WB · CLAHE<br/>edge-gated unsharp mask · QA · artifact fix"]
        VER["Verifier (engine.verify)<br/>prompts.py parser → OWL-ViT boxes → relation checks"]
        CLIP[CLIP score]
    end

    subgraph Offline["Offline — eval/"]
        TR[train_lora.py<br/>LoRA on COCO train split]
        EV[evaluate.py<br/>CLIP · FID · KID · latency · memory · size<br/>ablation · failure cases]
        RES[(results/results.json<br/>eval_P3_TEAM.md)]
    end

    UI --> AD --> WS
    WS --> ENC --> Tiers --> DEC --> POST --> VER --> WS
    Tiers -. each step .-> PRV -. preview JPEG .-> WS
    POST --> CLIP --> WS
    WS --> VIEW
    TR -. custom LoRA fused at load .-> Q
    EV --> RES --> TI --> AD
```

## Request lifecycle

1. The browser estimates the device class (`navigator.deviceMemory`, cores, user agent) and maps it to a memory allowance. **Adaptive** picks the best tier whose measured peak memory fits; a forced tier that does not fit is downgraded before sending (resource-aware fallback, client side).
2. The server validates the request, waits for the GPU lock (one image at a time), and runs the tier.
3. Each denoising step's latents are decoded by TAESD (~1M params) into a 256 px JPEG and streamed, so the image visibly forms.
4. The final latents are decoded by the tier's decoder, then the OpenCV chain runs (`postprocess.run`).
5. Quality assessment computes sharpness, clipping, colourfulness and an 8-px checkerboard score; detected artifacts are corrected.
6. The prompt is parsed into objects, attributes and relations; OWL-ViT finds each object; relations are checked geometrically.
7. If CUDA runs out of memory, the server retries on the next lighter tier and tells the client (resource-aware fallback, server side).
8. The client shows the result, the metrics and the checks; if a check failed it retries once with a new seed.
