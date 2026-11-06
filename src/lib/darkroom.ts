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

export const DEFAULT_TIERS: Record<TierKey, TierSpec> = {
  baseline: {
    name: "Baseline",
    unet: "SD 1.5 UNet",
    steps: 50,
    res: 512,
    precision: "FP32",
    decoder: "Full VAE",
    model_mb: 4260,
    measured: { clip: 0.318, gen_ms_mean: 7700, peakMB: 6200, sizeMB: 4260, encMs: 110, stepMs: 150, decMs: 90 },
  },
  quality: {
    name: "Quality",
    unet: "SD 1.5 + LCM-LoRA",
    steps: 8,
    res: 512,
    precision: "FP16",
    decoder: "Full VAE",
    model_mb: 2130,
    measured: { clip: 0.312, gen_ms_mean: 510, peakMB: 2900, sizeMB: 2130, encMs: 32, stepMs: 48, decMs: 56 },
  },
  balanced: {
    name: "Balanced",
    unet: "SD 1.5 + LCM-LoRA",
    steps: 4,
    res: 512,
    precision: "FP16",
    decoder: "TAESD",
    model_mb: 1970,
    measured: { clip: 0.304, gen_ms_mean: 240, peakMB: 2400, sizeMB: 1970, encMs: 20, stepMs: 28, decMs: 40 },
  },
  lite: {
    name: "Lite",
    unet: "SD 1.5 + LCM-LoRA INT8",
    steps: 3,
    res: 384,
    precision: "INT8",
    decoder: "TAESD",
    model_mb: 1110,
    measured: { clip: 0.289, gen_ms_mean: 230, peakMB: 1500, sizeMB: 1110, encMs: 18, stepMs: 30, decMs: 38 },
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

export type SceneObject = { label: string; box: [number, number, number, number]; score: number; ok: boolean };
export type DemoScene = { frames: string[]; raw: string; processed: string; detections: SceneObject[]; checks: { text: string; ok: boolean }[] };

const COLOR_MAP: Record<string, string> = {
  red: "#ed554d", blue: "#478cd9", green: "#58ab78", yellow: "#f5c94c", orange: "#ec8844",
  purple: "#9c6cc3", brown: "#805b49", white: "#e6e3dd", black: "#292b34", pink: "#dc8da7",
  teal: "#37aaa3", silver: "#aab4be",
};
const OBJECTS = ["cube", "sphere", "cup", "book", "cat", "car", "rickshaw", "temple", "lamp", "moon"];
const OBJECT_COLOR: Record<string, string> = {
  cube: "#ec554d", sphere: "#448bd8", cup: "#61a77b", book: "#b47a4c", cat: "#dfa04e",
  car: "#438bc0", rickshaw: "#e3be49", temple: "#dfd9c7", lamp: "#d9a64c", moon: "#ece3b9",
};

function colorFromPrompt(prompt: string, noun: string, index: number): string {
  const colors = prompt.toLowerCase().match(/\b(red|blue|green|yellow|orange|purple|brown|white|black|pink|teal|silver)\b/g) ?? [];
  const preceding = prompt.toLowerCase().slice(0, Math.max(0, prompt.toLowerCase().indexOf(noun)));
  const lastColor = [...preceding.matchAll(/\b(red|blue|green|yellow|orange|purple|brown|white|black|pink|teal|silver)\b/g)].at(-1)?.[1];
  return (lastColor && COLOR_MAP[lastColor]) || COLOR_MAP[colors[index]] || OBJECT_COLOR[noun] || "#c8a779";
}

function objectPosition(prompt: string, index: number, total: number): { x: number; y: number } {
  const lower = prompt.toLowerCase();
  const x = lower.includes("left of") || lower.includes("to the left") ? index === 0 ? 0.31 : 0.69
    : lower.includes("right of") || lower.includes("to the right") ? index === 0 ? 0.69 : 0.31
      : total === 1 ? 0.5 : 0.28 + (index / Math.max(1, total - 1)) * 0.44;
  const y = lower.includes("on top") || lower.includes("above") ? index === 0 ? 0.53 : 0.72
    : lower.includes("below") ? index === 0 ? 0.68 : 0.5
      : lower.includes("in front") || lower.includes("beside") ? index === 0 ? 0.7 : 0.53 : 0.62;
  return { x, y };
}

function drawObject(ctx: CanvasRenderingContext2D, noun: string, color: string, cx: number, cy: number, size: number): void {
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
    const ball = ctx.createRadialGradient(-size * .35, -size * .4, 2, 0, 0, size);
    ball.addColorStop(0, "#fff6d7"); ball.addColorStop(.18, color); ball.addColorStop(1, "#302c35");
    ctx.fillStyle = ball;
    ctx.beginPath(); ctx.arc(0, 0, size, 0, Math.PI * 2); ctx.fill();
  } else if (noun === "cube" || noun === "book") {
    const w = size * 1.6; const h = size * 1.35;
    ctx.beginPath(); ctx.moveTo(-w / 2, -h / 2); ctx.lineTo(0, -h / 2 - size * .35); ctx.lineTo(w / 2, -h / 2); ctx.lineTo(0, -h / 2 + size * .35); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-w / 2, -h / 2); ctx.lineTo(0, -h / 2 + size * .35); ctx.lineTo(0, h / 2); ctx.lineTo(-w / 2, h / 2 - size * .35); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -h / 2 + size * .35); ctx.lineTo(w / 2, -h / 2); ctx.lineTo(w / 2, h / 2 - size * .35); ctx.lineTo(0, h / 2); ctx.closePath(); ctx.fill(); ctx.stroke();
    if (noun === "book") { ctx.shadowColor = "transparent"; ctx.strokeStyle = "rgba(255,255,255,.42)"; ctx.beginPath(); ctx.moveTo(-w / 2 + 10, -h / 2 + 8); ctx.lineTo(-w / 2 + 10, h / 2 - 10); ctx.stroke(); }
  } else if (noun === "cup") {
    ctx.beginPath(); ctx.moveTo(-size * .65, -size * .55); ctx.lineTo(size * .65, -size * .55); ctx.lineTo(size * .48, size * .8); ctx.quadraticCurveTo(0, size, -size * .48, size * .8); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(0, -size * .55, size * .65, size * .19, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(size * .56, 0, size * .38, -1.2, 1.2); ctx.stroke();
  } else if (noun === "cat") {
    ctx.beginPath(); ctx.ellipse(0, size * .45, size * .8, size * .55, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(0, -size * .35, size * .55, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.moveTo(-size * .42, -size * .65); ctx.lineTo(-size * .48, -size * 1.2); ctx.lineTo(-size * .05, -size * .78); ctx.fill();
    ctx.beginPath(); ctx.moveTo(size * .42, -size * .65); ctx.lineTo(size * .48, -size * 1.2); ctx.lineTo(size * .05, -size * .78); ctx.fill();
    ctx.shadowColor = "transparent"; ctx.fillStyle = "#282630"; ctx.beginPath(); ctx.arc(-size * .2, -size * .4, 3, 0, Math.PI * 2); ctx.arc(size * .2, -size * .4, 3, 0, Math.PI * 2); ctx.fill();
  } else if (noun === "lamp") {
    ctx.fillRect(-size * .12, -size * .1, size * .24, size * 1.4);
    ctx.beginPath(); ctx.moveTo(-size * .7, -size * .15); ctx.lineTo(-size * .4, -size * .9); ctx.lineTo(size * .4, -size * .9); ctx.lineTo(size * .7, -size * .15); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.ellipse(0, size * 1.3, size * .55, size * .16, 0, 0, Math.PI * 2); ctx.fill();
  } else if (noun === "temple") {
    ctx.fillRect(-size * .75, -size * .2, size * 1.5, size * .95);
    ctx.beginPath(); ctx.moveTo(-size, -size * .2); ctx.lineTo(0, -size * 1.1); ctx.lineTo(size, -size * .2); ctx.closePath(); ctx.fill();
    ctx.shadowColor = "transparent"; ctx.fillStyle = "#2c2931"; ctx.fillRect(-size * .16, size * .2, size * .32, size * .55);
  } else {
    ctx.beginPath(); ctx.roundRect(-size, -size * .35, size * 2, size * .85, size * .25); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(-size * .6, size * .55, size * .22, 0, Math.PI * 2); ctx.arc(size * .6, size * .55, size * .22, 0, Math.PI * 2); ctx.fill();
    if (noun === "rickshaw") { ctx.shadowColor = "transparent"; ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.beginPath(); ctx.moveTo(-size * .7, -size * .35); ctx.quadraticCurveTo(0, -size * 1.2, size * .7, -size * .35); ctx.stroke(); }
  }
  ctx.restore();
}

export function makeDemoScene(prompt: string, steps: number, seed: number, post: { contrast: boolean; wb: boolean; sharp: boolean; upscale: boolean }): DemoScene {
  const dimension = post.upscale ? 512 : 448;
  const nouns = OBJECTS.filter((noun) => prompt.toLowerCase().includes(noun));
  if (nouns.length === 0) nouns.push("sphere", "cube");
  const canvas = document.createElement("canvas");
  canvas.width = dimension; canvas.height = dimension;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { frames: [], raw: "", processed: "", detections: [], checks: [] };
  const lower = prompt.toLowerCase();
  const sky = ctx.createLinearGradient(0, 0, 0, dimension);
  if (lower.includes("night")) { sky.addColorStop(0, "#171a31"); sky.addColorStop(.57, "#433047"); sky.addColorStop(1, "#b46b57"); }
  else if (lower.includes("sunset") || lower.includes("sunrise")) { sky.addColorStop(0, "#503653"); sky.addColorStop(.48, "#df875f"); sky.addColorStop(1, "#efc086"); }
  else if (lower.includes("desert")) { sky.addColorStop(0, "#6e8f96"); sky.addColorStop(.58, "#dbb982"); sky.addColorStop(1, "#8f654c"); }
  else { sky.addColorStop(0, "#8499a1"); sky.addColorStop(.64, "#d5b999"); sky.addColorStop(1, "#62554c"); }
  ctx.fillStyle = sky; ctx.fillRect(0, 0, dimension, dimension);
  const seedRand = (n: number) => { const x = Math.sin(n * 127.1 + seed * 311.7) * 43758.5453; return x - Math.floor(x); };
  for (let i = 0; i < 22; i++) {
    const x = seedRand(i + 4) * dimension; const y = seedRand(i + 40) * dimension * .54;
    ctx.fillStyle = `rgba(255,235,201,${seedRand(i + 90) * .45})`; ctx.beginPath(); ctx.arc(x, y, seedRand(i + 11) * 2 + .4, 0, Math.PI * 2); ctx.fill();
  }
  const sunY = lower.includes("night") ? dimension * .24 : dimension * .31;
  ctx.fillStyle = lower.includes("night") ? "rgba(245,229,198,.78)" : "rgba(255,207,130,.68)";
  ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 40; ctx.beginPath(); ctx.arc(dimension * .76, sunY, dimension * .085, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
  const horizon = dimension * .68;
  ctx.fillStyle = "rgba(39,38,44,.22)"; ctx.fillRect(0, horizon, dimension, dimension - horizon);
  const tabletop = ctx.createLinearGradient(0, horizon, 0, dimension);
  tabletop.addColorStop(0, lower.includes("wood") ? "#795943" : "#6c625d"); tabletop.addColorStop(1, "#29282d");
  ctx.fillStyle = tabletop; ctx.fillRect(0, horizon, dimension, dimension - horizon);
  for (let i = 0; i < 13; i++) { ctx.strokeStyle = `rgba(235,201,158,${.025 + seedRand(i + 120) * .045})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, horizon + i * 12); ctx.bezierCurveTo(dimension * .4, horizon + i * 12 + 9, dimension * .72, horizon + i * 12 - 7, dimension, horizon + i * 12 + 2); ctx.stroke(); }
  const detections: SceneObject[] = [];
  nouns.forEach((noun, index) => {
    const point = objectPosition(prompt, index, nouns.length);
    const size = dimension * (nouns.length > 2 ? .085 : .115);
    const x = dimension * point.x; const y = dimension * point.y;
    drawObject(ctx, noun, colorFromPrompt(prompt, noun, index), x, y, size);
    detections.push({ label: noun, box: [Math.max(0, point.x - .14), Math.max(0, point.y - .2), Math.min(1, point.x + .14), Math.min(1, point.y + .14)], score: .89 + seedRand(index + 2) * .1, ok: true });
  });
  const raw = canvas.toDataURL("image/jpeg", .91);
  const processedCanvas = document.createElement("canvas"); processedCanvas.width = dimension; processedCanvas.height = dimension;
  const pctx = processedCanvas.getContext("2d");
  if (pctx) {
    pctx.filter = `${post.contrast ? "contrast(1.08) saturate(1.04) " : ""}${post.wb ? "sepia(.04) " : ""}${post.sharp ? "contrast(1.08) " : ""}brightness(1.015)`;
    pctx.drawImage(canvas, 0, 0);
  }
  const processed = processedCanvas.toDataURL("image/jpeg", .94);
  const noise = document.createElement("canvas"); noise.width = 48; noise.height = 48;
  const nctx = noise.getContext("2d");
  if (nctx) {
    const noiseImage = nctx.createImageData(48, 48);
    for (let i = 0; i < noiseImage.data.length; i += 4) {
      const value = Math.floor(seedRand(i / 4 + 19) * 255);
      noiseImage.data[i] = value; noiseImage.data[i + 1] = Math.floor(value * .92); noiseImage.data[i + 2] = 255 - value; noiseImage.data[i + 3] = 255;
    }
    nctx.putImageData(noiseImage, 0, 0);
  }
  const image = new Image(); image.src = processed;
  const frames: string[] = [];
  for (let i = 0; i < steps; i++) {
    const frame = document.createElement("canvas"); frame.width = dimension; frame.height = dimension;
    const fctx = frame.getContext("2d");
    if (!fctx) continue;
    fctx.filter = `blur(${Math.round(((steps - i - 1) / Math.max(1, steps)) * 15)}px)`;
    fctx.drawImage(processedCanvas, 0, 0);
    fctx.filter = "none";
    if (nctx) { fctx.globalAlpha = .34 * (1 - i / Math.max(1, steps)); fctx.drawImage(noise, 0, 0, dimension, dimension); fctx.globalAlpha = 1; }
    frames.push(frame.toDataURL("image/jpeg", .84));
  }
  const checks = nouns.map((noun) => ({ text: `“${noun}” appears in the scene`, ok: true }));
  checks.push({ text: lower.includes("left of") ? "Objects follow the requested left-to-right order" : "Scene composition follows the prompt", ok: true });
  return { frames, raw, processed, detections, checks };
}

export function getBackendBase(wsUrl: string): string {
  return wsUrl.replace(/^wss:/i, "https:").replace(/^ws:/i, "http:").replace(/\/ws\/?$/i, "").replace(/\/$/, "");
}