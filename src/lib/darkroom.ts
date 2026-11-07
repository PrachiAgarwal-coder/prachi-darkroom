export type TierKey = "baseline" | "quality" | "balanced" | "lite";
export type DeviceKey = "auto" | "budget" | "mid" | "flagship";

export type TierSpec = {
  name: string;
  unet: string;
  steps: number;
  res: number;
  precision: string;
  decoder: string;
  model_mb: number;
  measured: {
    clip: number;
    gen_ms_mean: number;
    peakMB: number;
    sizeMB: number;
    encMs: number;
    stepMs: number;
    decMs: number;
  };
};

// Latency, CLIP and size: measured on a Tesla T4 (eval/results/eval_P3_T03.md).
// peakMB: measured per tier in isolation (eval/measure_memory.py).
export const DEFAULT_TIERS: Record<TierKey, TierSpec> = {
  baseline: {
    name: "Baseline",
    unet: "SD 1.5 UNet",
    steps: 50,
    res: 512,
    precision: "FP32",
    decoder: "Full VAE",
    model_mb: 4265,
    measured: {
      clip: 0.308,
      gen_ms_mean: 31815,
      peakMB: 5111,
      sizeMB: 4265,
      encMs: 19,
      stepMs: 617,
      decMs: 958,
    },
  },
  quality: {
    name: "Quality",
    unet: "SD 1.5 + LCM-LoRA",
    steps: 8,
    res: 512,
    precision: "FP16",
    decoder: "Full VAE",
    model_mb: 2132,
    measured: {
      clip: 0.299,
      gen_ms_mean: 951,
      peakMB: 2810,
      sizeMB: 2132,
      encMs: 9,
      stepMs: 93,
      decMs: 195,
    },
  },
  balanced: {
    name: "Balanced",
    unet: "SD 1.5 + LCM-LoRA",
    steps: 4,
    res: 512,
    precision: "FP16",
    decoder: "TAESD",
    model_mb: 1970,
    measured: {
      clip: 0.301,
      gen_ms_mean: 402,
      peakMB: 2168,
      sizeMB: 1970,
      encMs: 10,
      stepMs: 93,
      decMs: 21,
    },
  },
  lite: {
    name: "Lite",
    unet: "SD 1.5 + LCM-LoRA INT8",
    steps: 3,
    res: 384,
    precision: "INT8",
    decoder: "TAESD",
    model_mb: 1112,
    measured: {
      clip: 0.293,
      gen_ms_mean: 348,
      peakMB: 1276,
      sizeMB: 1112,
      encMs: 10,
      stepMs: 109,
      decMs: 13,
    },
  },
};

export const DEVICE_ALLOWANCE: Record<Exclude<DeviceKey, "auto">, number> = {
  budget: 1800,
  mid: 2600,
  flagship: 4200,
};

export const DEVICE_LABEL: Record<DeviceKey, string> = {
  auto: "Auto",
  budget: "Budget phone",
  mid: "Mid-range",
  flagship: "Flagship",
};

export const TIERS: TierKey[] = ["quality", "balanced", "lite"];
export const TIER_LABEL: Record<TierKey, string> = {
  baseline: "Baseline",
  quality: "Quality",
  balanced: "Balanced",
  lite: "Lite",
};

export function resolveDevice(device: DeviceKey, deviceMemory = 6): Exclude<DeviceKey, "auto"> {
  if (device !== "auto") return device;
  if (deviceMemory <= 3) return "budget";
  if (deviceMemory <= 6) return "mid";
  return "flagship";
}

export function resolveTier(
  requested: TierKey | "adaptive",
  allowance: number,
  tiers: Record<TierKey, TierSpec>,
): { tier: TierKey; downgraded: boolean; requested?: TierKey } {
  if (requested === "adaptive") {
    const fit = TIERS.find((key) => tiers[key].measured.peakMB <= allowance);
    return { tier: fit ?? "lite", downgraded: false };
  }
  if (tiers[requested].measured.peakMB <= allowance) return { tier: requested, downgraded: false };
  const fit = TIERS.find((key) => tiers[key].measured.peakMB <= allowance);
  return { tier: fit ?? "lite", downgraded: true, requested };
}

export type SceneObject = {
  label: string;
  box: [number, number, number, number];
  score: number;
  ok: boolean;
};
export type DemoScene = {
  frames: string[];
  raw: string;
  processed: string;
  detections: SceneObject[];
  checks: { text: string; ok: boolean }[];
};

const COLOR_MAP: Record<string, string> = {
  red: "#ed554d",
  blue: "#478cd9",
  green: "#58ab78",
  yellow: "#f5c94c",
  orange: "#ec8844",
  purple: "#9c6cc3",
  brown: "#805b49",
  white: "#e6e3dd",
  black: "#292b34",
  pink: "#dc8da7",
  teal: "#37aaa3",
  silver: "#aab4be",
};
const OBJECTS = [
  "cube",
  "sphere",
  "cup",
  "book",
  "cat",
  "car",
  "rickshaw",
  "temple",
  "lamp",
  "moon",
];
const OBJECT_COLOR: Record<string, string> = {
  cube: "#ec554d",
  sphere: "#448bd8",
  cup: "#61a77b",
  book: "#b47a4c",
  cat: "#dfa04e",
  car: "#438bc0",
  rickshaw: "#e3be49",
  temple: "#dfd9c7",
  lamp: "#d9a64c",
  moon: "#ece3b9",
};

function colorFromPrompt(prompt: string, noun: string, index: number): string {
  const colors =
    prompt
      .toLowerCase()
      .match(/\b(red|blue|green|yellow|orange|purple|brown|white|black|pink|teal|silver)\b/g) ?? [];
  const preceding = prompt.toLowerCase().slice(0, Math.max(0, prompt.toLowerCase().indexOf(noun)));
  const lastColor = [
    ...preceding.matchAll(
      /\b(red|blue|green|yellow|orange|purple|brown|white|black|pink|teal|silver)\b/g,
    ),
  ].at(-1)?.[1];
  const indexed = colors[index];
  return (
    (lastColor && COLOR_MAP[lastColor]) ||
    (indexed && COLOR_MAP[indexed]) ||
    OBJECT_COLOR[noun] ||
    "#c8a779"
  );
}

function objectPosition(prompt: string, index: number, total: number): { x: number; y: number } {
  const lower = prompt.toLowerCase();
  const x =
    lower.includes("left of") || lower.includes("to the left")
      ? index === 0
        ? 0.31
        : 0.69
      : lower.includes("right of") || lower.includes("to the right")
        ? index === 0
          ? 0.69
          : 0.31
        : total === 1
          ? 0.5
          : 0.28 + (index / Math.max(1, total - 1)) * 0.44;
  const y =
    lower.includes("on top") || lower.includes("above")
      ? index === 0
        ? 0.53
        : 0.72
      : lower.includes("below")
        ? index === 0
          ? 0.68
          : 0.5
        : lower.includes("in front") || lower.includes("beside")
          ? index === 0
            ? 0.7
            : 0.53
          : 0.62;
  return { x, y };
}

function drawObject(
  ctx: CanvasRenderingContext2D,
  noun: string,
  color: string,
  cx: number,
  cy: number,
  size: number,
): void {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.shadowColor = "rgba(9, 8, 12, 0.35)";
  ctx.shadowBlur = 22;
  ctx.shadowOffsetY = 15;
  const gradient = ctx.createLinearGradient(-size, -size, size, size);
  gradient.addColorStop(0, color);
  gradient.addColorStop(1, "#24212b");
  ctx.fillStyle = gradient;
  ctx.strokeStyle = "rgba(255,255,255,.28)";
  ctx.lineWidth = 2;
  if (noun === "sphere" || noun === "moon") {
    const ball = ctx.createRadialGradient(-size * 0.35, -size * 0.4, 2, 0, 0, size);
    ball.addColorStop(0, "#fff6d7");
    ball.addColorStop(0.18, color);
    ball.addColorStop(1, "#302c35");
    ctx.fillStyle = ball;
    ctx.beginPath();
    ctx.arc(0, 0, size, 0, Math.PI * 2);
    ctx.fill();
  } else if (noun === "cube" || noun === "book") {
    const w = size * 1.6;
    const h = size * 1.35;
    ctx.beginPath();
    ctx.moveTo(-w / 2, -h / 2);
    ctx.lineTo(0, -h / 2 - size * 0.35);
    ctx.lineTo(w / 2, -h / 2);
    ctx.lineTo(0, -h / 2 + size * 0.35);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-w / 2, -h / 2);
    ctx.lineTo(0, -h / 2 + size * 0.35);
    ctx.lineTo(0, h / 2);
    ctx.lineTo(-w / 2, h / 2 - size * 0.35);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, -h / 2 + size * 0.35);
    ctx.lineTo(w / 2, -h / 2);
    ctx.lineTo(w / 2, h / 2 - size * 0.35);
    ctx.lineTo(0, h / 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    if (noun === "book") {
      ctx.shadowColor = "transparent";
      ctx.strokeStyle = "rgba(255,255,255,.42)";
      ctx.beginPath();
      ctx.moveTo(-w / 2 + 10, -h / 2 + 8);
      ctx.lineTo(-w / 2 + 10, h / 2 - 10);
      ctx.stroke();
    }
  } else if (noun === "cup") {
    ctx.beginPath();
    ctx.moveTo(-size * 0.65, -size * 0.55);
    ctx.lineTo(size * 0.65, -size * 0.55);
    ctx.lineTo(size * 0.48, size * 0.8);
    ctx.quadraticCurveTo(0, size, -size * 0.48, size * 0.8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(0, -size * 0.55, size * 0.65, size * 0.19, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(size * 0.56, 0, size * 0.38, -1.2, 1.2);
    ctx.stroke();
  } else if (noun === "cat") {
    ctx.beginPath();
    ctx.ellipse(0, size * 0.45, size * 0.8, size * 0.55, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, -size * 0.35, size * 0.55, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-size * 0.42, -size * 0.65);
    ctx.lineTo(-size * 0.48, -size * 1.2);
    ctx.lineTo(-size * 0.05, -size * 0.78);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(size * 0.42, -size * 0.65);
    ctx.lineTo(size * 0.48, -size * 1.2);
    ctx.lineTo(size * 0.05, -size * 0.78);
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#282630";
    ctx.beginPath();
    ctx.arc(-size * 0.2, -size * 0.4, 3, 0, Math.PI * 2);
    ctx.arc(size * 0.2, -size * 0.4, 3, 0, Math.PI * 2);
    ctx.fill();
  } else if (noun === "lamp") {
    ctx.fillRect(-size * 0.12, -size * 0.1, size * 0.24, size * 1.4);
    ctx.beginPath();
    ctx.moveTo(-size * 0.7, -size * 0.15);
    ctx.lineTo(-size * 0.4, -size * 0.9);
    ctx.lineTo(size * 0.4, -size * 0.9);
    ctx.lineTo(size * 0.7, -size * 0.15);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(0, size * 1.3, size * 0.55, size * 0.16, 0, 0, Math.PI * 2);
    ctx.fill();
  } else if (noun === "temple") {
    ctx.fillRect(-size * 0.75, -size * 0.2, size * 1.5, size * 0.95);
    ctx.beginPath();
    ctx.moveTo(-size, -size * 0.2);
    ctx.lineTo(0, -size * 1.1);
    ctx.lineTo(size, -size * 0.2);
    ctx.closePath();
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.fillStyle = "#2c2931";
    ctx.fillRect(-size * 0.16, size * 0.2, size * 0.32, size * 0.55);
  } else {
    ctx.beginPath();
    ctx.roundRect(-size, -size * 0.35, size * 2, size * 0.85, size * 0.25);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(-size * 0.6, size * 0.55, size * 0.22, 0, Math.PI * 2);
    ctx.arc(size * 0.6, size * 0.55, size * 0.22, 0, Math.PI * 2);
    ctx.fill();
    if (noun === "rickshaw") {
      ctx.shadowColor = "transparent";
      ctx.strokeStyle = "rgba(255,255,255,.5)";
      ctx.beginPath();
      ctx.moveTo(-size * 0.7, -size * 0.35);
      ctx.quadraticCurveTo(0, -size * 1.2, size * 0.7, -size * 0.35);
      ctx.stroke();
    }
  }
  ctx.restore();
}

export function makeDemoScene(
  prompt: string,
  steps: number,
  seed: number,
  post: { contrast: boolean; wb: boolean; sharp: boolean; upscale: boolean },
): DemoScene {
  const dimension = post.upscale ? 512 : 448;
  const nouns = OBJECTS.filter((noun) => prompt.toLowerCase().includes(noun));
  if (nouns.length === 0) nouns.push("sphere", "cube");
  const canvas = document.createElement("canvas");
  canvas.width = dimension;
  canvas.height = dimension;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { frames: [], raw: "", processed: "", detections: [], checks: [] };
  const lower = prompt.toLowerCase();
  const sky = ctx.createLinearGradient(0, 0, 0, dimension);
  if (lower.includes("night")) {
    sky.addColorStop(0, "#171a31");
    sky.addColorStop(0.57, "#433047");
    sky.addColorStop(1, "#b46b57");
  } else if (lower.includes("sunset") || lower.includes("sunrise")) {
    sky.addColorStop(0, "#503653");
    sky.addColorStop(0.48, "#df875f");
    sky.addColorStop(1, "#efc086");
  } else if (lower.includes("desert")) {
    sky.addColorStop(0, "#6e8f96");
    sky.addColorStop(0.58, "#dbb982");
    sky.addColorStop(1, "#8f654c");
  } else {
    sky.addColorStop(0, "#8499a1");
    sky.addColorStop(0.64, "#d5b999");
    sky.addColorStop(1, "#62554c");
  }
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, dimension, dimension);
  const seedRand = (n: number) => {
    const x = Math.sin(n * 127.1 + seed * 311.7) * 43758.5453;
    return x - Math.floor(x);
  };
  for (let i = 0; i < 22; i++) {
    const x = seedRand(i + 4) * dimension;
    const y = seedRand(i + 40) * dimension * 0.54;
    ctx.fillStyle = `rgba(255,235,201,${seedRand(i + 90) * 0.45})`;
    ctx.beginPath();
    ctx.arc(x, y, seedRand(i + 11) * 2 + 0.4, 0, Math.PI * 2);
    ctx.fill();
  }
  const sunY = lower.includes("night") ? dimension * 0.24 : dimension * 0.31;
  ctx.fillStyle = lower.includes("night") ? "rgba(245,229,198,.78)" : "rgba(255,207,130,.68)";
  ctx.shadowColor = ctx.fillStyle;
  ctx.shadowBlur = 40;
  ctx.beginPath();
  ctx.arc(dimension * 0.76, sunY, dimension * 0.085, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  const horizon = dimension * 0.68;
  ctx.fillStyle = "rgba(39,38,44,.22)";
  ctx.fillRect(0, horizon, dimension, dimension - horizon);
  const tabletop = ctx.createLinearGradient(0, horizon, 0, dimension);
  tabletop.addColorStop(0, lower.includes("wood") ? "#795943" : "#6c625d");
  tabletop.addColorStop(1, "#29282d");
  ctx.fillStyle = tabletop;
  ctx.fillRect(0, horizon, dimension, dimension - horizon);
  for (let i = 0; i < 13; i++) {
    ctx.strokeStyle = `rgba(235,201,158,${0.025 + seedRand(i + 120) * 0.045})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, horizon + i * 12);
    ctx.bezierCurveTo(
      dimension * 0.4,
      horizon + i * 12 + 9,
      dimension * 0.72,
      horizon + i * 12 - 7,
      dimension,
      horizon + i * 12 + 2,
    );
    ctx.stroke();
  }
  const detections: SceneObject[] = [];
  nouns.forEach((noun, index) => {
    const point = objectPosition(prompt, index, nouns.length);
    const size = dimension * (nouns.length > 2 ? 0.085 : 0.115);
    const x = dimension * point.x;
    const y = dimension * point.y;
    drawObject(ctx, noun, colorFromPrompt(prompt, noun, index), x, y, size);
    detections.push({
      label: noun,
      box: [
        Math.max(0, point.x - 0.14),
        Math.max(0, point.y - 0.2),
        Math.min(1, point.x + 0.14),
        Math.min(1, point.y + 0.14),
      ],
      score: 0.89 + seedRand(index + 2) * 0.1,
      ok: true,
    });
  });
  const raw = canvas.toDataURL("image/jpeg", 0.91);
  const processedCanvas = document.createElement("canvas");
  processedCanvas.width = dimension;
  processedCanvas.height = dimension;
  const pctx = processedCanvas.getContext("2d");
  if (pctx) {
    pctx.filter = `${post.contrast ? "contrast(1.08) saturate(1.04) " : ""}${post.wb ? "sepia(.04) " : ""}${post.sharp ? "contrast(1.08) " : ""}brightness(1.015)`;
    pctx.drawImage(canvas, 0, 0);
  }
  const processed = processedCanvas.toDataURL("image/jpeg", 0.94);
  const noise = document.createElement("canvas");
  noise.width = 48;
  noise.height = 48;
  const nctx = noise.getContext("2d");
  if (nctx) {
    const noiseImage = nctx.createImageData(48, 48);
    for (let i = 0; i < noiseImage.data.length; i += 4) {
      const value = Math.floor(seedRand(i / 4 + 19) * 255);
      noiseImage.data[i] = value;
      noiseImage.data[i + 1] = Math.floor(value * 0.92);
      noiseImage.data[i + 2] = 255 - value;
      noiseImage.data[i + 3] = 255;
    }
    nctx.putImageData(noiseImage, 0, 0);
  }
  const image = new Image();
  image.src = processed;
  const frames: string[] = [];
  for (let i = 0; i < steps; i++) {
    const frame = document.createElement("canvas");
    frame.width = dimension;
    frame.height = dimension;
    const fctx = frame.getContext("2d");
    if (!fctx) continue;
    fctx.filter = `blur(${Math.round(((steps - i - 1) / Math.max(1, steps)) * 15)}px)`;
    fctx.drawImage(processedCanvas, 0, 0);
    fctx.filter = "none";
    if (nctx) {
      fctx.globalAlpha = 0.34 * (1 - i / Math.max(1, steps));
      fctx.drawImage(noise, 0, 0, dimension, dimension);
      fctx.globalAlpha = 1;
    }
    frames.push(frame.toDataURL("image/jpeg", 0.84));
  }
  const checks = nouns.map((noun) => ({ text: `“${noun}” appears in the scene`, ok: true }));
  checks.push({
    text: lower.includes("left of")
      ? "Objects follow the requested left-to-right order"
      : "Scene composition follows the prompt",
    ok: true,
  });
  return { frames, raw, processed, detections, checks };
}

export function getBackendBase(wsUrl: string): string {
  return wsUrl
    .replace(/^wss:/i, "https:")
    .replace(/^ws:/i, "http:")
    .replace(/\/ws\/?$/i, "")
    .replace(/\/$/, "");
}
/** Map the server's GET /tiers payload onto TierSpec. The server sends measured
 *  numbers as flat keys (clip, gen_ms_mean, peakMB, sizeMB, encMs, stepMs, decMs);
 *  a nested `measured` object is accepted too. Missing tiers keep their defaults. */
export function normalizeTiers(payload: unknown): Record<TierKey, TierSpec> {
  const next = structuredClone(DEFAULT_TIERS);
  if (!payload || typeof payload !== "object") return next;
  for (const key of ["baseline", "quality", "balanced", "lite"] as TierKey[]) {
    type Raw = Partial<
      Record<
        | "name"
        | "unet"
        | "precision"
        | "decoder"
        | "steps"
        | "res"
        | "model_mb"
        | "clip"
        | "gen_ms_mean"
        | "peakMB"
        | "sizeMB"
        | "encMs"
        | "stepMs"
        | "decMs",
        unknown
      >
    > & { measured?: Raw };
    const raw = (payload as Partial<Record<TierKey, Raw>>)[key];
    if (!raw || typeof raw !== "object") continue;
    const flat: Raw = { ...raw, ...(raw.measured ?? {}) };
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
    const t = next[key];
    if (typeof flat.name === "string") t.name = flat.name;
    if (typeof flat.unet === "string") t.unet = flat.unet;
    if (typeof flat.precision === "string") t.precision = flat.precision;
    if (typeof flat.decoder === "string") t.decoder = flat.decoder;
    t.steps = num(flat.steps) ?? t.steps;
    t.res = num(flat.res) ?? t.res;
    t.model_mb = num(flat.model_mb) ?? t.model_mb;
    const m = t.measured;
    m.clip = num(flat.clip) ?? m.clip;
    m.gen_ms_mean = num(flat.gen_ms_mean) ?? m.gen_ms_mean;
    m.peakMB = num(flat.peakMB) ?? m.peakMB;
    m.sizeMB = num(flat.sizeMB) ?? num(flat.model_mb) ?? m.sizeMB;
    m.encMs = num(flat.encMs) ?? m.encMs;
    m.stepMs = num(flat.stepMs) ?? m.stepMs;
    m.decMs = num(flat.decMs) ?? m.decMs;
    (t as TierSpec & { available?: boolean; isMeasured?: boolean }).available = true;
    (t as TierSpec & { isMeasured?: boolean }).isMeasured = num(flat.gen_ms_mean) !== undefined;
  }
  return next;
}

export type JobMessage = {
  type?: string;
  stage?: string;
  state?: string;
  ms?: number;
  step?: number;
  total?: number;
  preview?: string;
  message?: string;
  from?: string;
  image?: string;
  processed?: string;
  metrics?: unknown;
  qa?: Record<string, unknown>;
  detections?: unknown;
  checks?: unknown;
  tier_used?: string;
  fallback_tier?: string;
};

/** One generation over the WebSocket. Resolves with the `result` message,
 *  rejects on `error`, socket failure, or 45 s without any message. */
export function runJob(
  wsUrl: string,
  request: Record<string, unknown>,
  onMessage: (m: JobMessage) => void,
): { promise: Promise<JobMessage>; cancel: () => void } {
  let socket: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let settled = false;
  const promise = new Promise<JobMessage>((resolve, reject) => {
    const fail = (msg: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket?.close();
      reject(new Error(msg));
    };
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () =>
          fail(
            "No response from the image server for 45 seconds. It may be starting up; try again in a minute.",
          ),
        45_000,
      );
    };
    try {
      socket = new WebSocket(wsUrl.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:"));
    } catch {
      fail("That server address is not a valid WebSocket URL.");
      return;
    }
    arm();
    socket.onopen = () => socket?.send(JSON.stringify(request));
    socket.onerror = () => fail("Could not reach the image server.");
    socket.onclose = () => {
      if (!settled) fail("The image server closed the connection.");
    };
    socket.onmessage = (event) => {
      arm();
      let m: JobMessage;
      try {
        m = JSON.parse(String(event.data)) as JobMessage;
      } catch {
        fail("The image server sent an unreadable message.");
        return;
      }
      if (m.type === "error") {
        fail(typeof m.message === "string" ? m.message : "The image server reported an error.");
        return;
      }
      onMessage(m);
      if (m.type === "result") {
        settled = true;
        clearTimeout(timer);
        socket?.close();
        resolve(m);
      }
    };
  });
  return {
    promise,
    cancel: () => {
      settled = true;
      clearTimeout(timer);
      socket?.close();
    },
  };
}
