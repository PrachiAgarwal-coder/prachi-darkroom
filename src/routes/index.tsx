import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { AnimatePresence, motion, useMotionValue, useSpring } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowLeftRight,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Cpu,
  Crosshair,
  Gauge,
  ImageIcon,
  Layers3,
  Link2,
  LoaderCircle,
  Maximize2,
  MemoryStick,
  Moon,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  Send,
  Settings2,
  Sparkles,
  Square,
  Timer,
  WandSparkles,
  Wifi,
  X,
  XCircle,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Toaster } from "@/components/ui/sonner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  DEFAULT_TIERS,
  DEVICE_ALLOWANCE,
  DEVICE_LABEL,
  getBackendBase,
  makeDemoScene,
  normalizeTiers,
  resolveDevice,
  resolveTier,
  runJob,
  TIER_LABEL,
  TIERS,
  type DeviceKey,
  type JobMessage,
  type TierKey,
  type TierSpec,
} from "@/lib/darkroom";

type StageKey = "encode" | "denoise" | "decode" | "post" | "qa" | "verify";
type StageState = "wait" | "run" | "ok";
type StageTiming = { state: StageState; ms?: number };
type Check = { text: string; ok: boolean; muted?: boolean };
type Metrics = {
  clip: number | null;
  peak_mb: number | null;
  model_mb: number | null;
  encode_ms: number;
  denoise_ms: number;
  decode_ms: number;
  post_ms: number;
  qa_ms: number;
  verify_ms: number;
  total_ms: number;
};
type Detection = {
  label: string;
  box: [number, number, number, number];
  score: number;
  ok: boolean;
};
type ResultState = {
  raw: string;
  processed: string;
  metrics: Metrics;
  qa: Record<string, unknown>;
  detections: Detection[];
  checks: Check[];
  tier: TierKey;
};
type HistoryEntry = ResultState & { id: number; prompt: string; image: string };
type ConnectionState = "demo" | "connecting" | "live" | "unreachable";
type PostSettings = { contrast: boolean; wb: boolean; sharp: boolean; upscale: boolean };
type BackendMessage = JobMessage;

const SUGGESTIONS = [
  "a red cube to the left of a blue sphere on a wooden table at sunset",
  "a yellow auto rickshaw in front of a white temple at night",
  "a green cup on top of a brown book, studio lighting",
  "two orange cats beside a purple lamp",
];
const STAGES: StageKey[] = ["encode", "denoise", "decode", "post", "qa", "verify"];
const STAGE_LABEL: Record<StageKey, string> = {
  encode: "Encode",
  denoise: "Denoise",
  decode: "Decode",
  post: "Post-process",
  qa: "Quality check",
  verify: "Verify prompt",
};
const DEFAULT_POST: PostSettings = { contrast: true, wb: true, sharp: true, upscale: true }; // matches the server and the evaluation
const STAGE_TIMING_KEYS: Record<StageKey, keyof Metrics> = {
  encode: "encode_ms",
  denoise: "denoise_ms",
  decode: "decode_ms",
  post: "post_ms",
  qa: "qa_ms",
  verify: "verify_ms",
};
const METRICS_ZERO: Metrics = {
  clip: null,
  peak_mb: null,
  model_mb: null,
  encode_ms: 0,
  denoise_ms: 0,
  decode_ms: 0,
  post_ms: 0,
  qa_ms: 0,
  verify_ms: 0,
  total_ms: 0,
};
const DEFAULT_URL =
  (import.meta.env as ImportMetaEnv & { VITE_DARKROOM_WS?: string }).VITE_DARKROOM_WS ?? "";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Darkroom — On-device image generation" },
      {
        name: "description",
        content: "A cinematic on-device text-to-image darkroom, built for phones.",
      },
      { property: "og:title", content: "Darkroom — On-device image generation" },
      {
        property: "og:description",
        content: "Bring a scene to life in a few optimized diffusion steps.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
    links: [
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap",
      },
    ],
  }),
  component: Darkroom,
});

function LatentField({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const points = Array.from({ length: window.innerWidth < 640 ? 180 : 340 }, (_, i) => ({
      x: (i * 83.71) % 1,
      y: (i * 47.17) % 1,
      phase: i * 0.83,
      radius: 0.45 + ((i * 13) % 10) * 0.12,
      hue: [275, 18, 42, 190][i % 4],
    }));
    let frame = 0;
    let raf = 0;
    const paint = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      if (
        canvas.width !== Math.round(width * ratio) ||
        canvas.height !== Math.round(height * ratio)
      ) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      for (const point of points) {
        const drift = reduced ? 0 : frame * 0.00018;
        const x = (point.x + Math.sin(point.phase + drift) * 0.016 + 1) % 1;
        const y = (point.y + Math.cos(point.phase * 0.7 + drift) * 0.021 + 1) % 1;
        const targetX = width * 0.5;
        const targetY = height * 0.42;
        const rush = activeRef.current ? 0.58 + Math.sin(frame * 0.035 + point.phase) * 0.035 : 0;
        const px = width * x * (1 - rush) + targetX * rush;
        const py = height * y * (1 - rush) + targetY * rush;
        ctx.fillStyle = `hsla(${point.hue}, 86%, ${activeRef.current ? 66 : 60}%, ${activeRef.current ? 0.4 : 0.2})`;
        ctx.beginPath();
        ctx.arc(px, py, point.radius * (activeRef.current ? 1.4 : 1), 0, Math.PI * 2);
        ctx.fill();
      }
      frame += 1;
      if (!reduced) raf = window.requestAnimationFrame(paint);
    };
    paint();
    const resize = () => {
      if (reduced) paint();
    };
    window.addEventListener("resize", resize);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);
  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0 h-full w-full opacity-75"
      aria-hidden
    />
  );
}

function CountValue({ value, digits = 0 }: { value: number; digits?: number }) {
  const motionValue = useMotionValue(value);
  const spring = useSpring(motionValue, { damping: 24, stiffness: 120, mass: 0.65 });
  const [display, setDisplay] = useState(value.toFixed(digits));
  useEffect(() => {
    motionValue.set(value);
  }, [motionValue, value]);
  useEffect(() => spring.on("change", (n) => setDisplay(n.toFixed(digits))), [spring, digits]);
  return <span className="tabular-nums">{display}</span>;
}

function IconButton({
  label,
  children,
  onClick,
}: {
  label: string;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <Button
      aria-label={label}
      title={label}
      variant="ghost"
      size="icon"
      className="h-9 w-9 border border-darkroom-panel-edge text-muted-foreground hover:text-foreground"
      onClick={onClick}
    >
      {children}
    </Button>
  );
}

function Metric({
  label,
  value,
  suffix,
  detail,
  icon: Icon,
  color = "text-foreground",
  digits = 0,
}: {
  label: string;
  value: number | null;
  suffix?: string;
  detail?: string;
  icon: typeof Activity;
  color?: string;
  digits?: number;
}) {
  return (
    <div className="min-h-[104px] rounded-xl border border-darkroom-panel-edge bg-darkroom-panel p-3 backdrop-blur-xl sm:p-4">
      <div className="flex items-center justify-between gap-2 text-[10px] font-medium uppercase tracking-[.14em] text-muted-foreground">
        <span>{label}</span>
        <Icon className="h-3.5 w-3.5 opacity-60" />
      </div>
      <div className={`mt-2 font-display text-2xl font-semibold tabular-nums ${color}`}>
        {value === null || !Number.isFinite(value) ? (
          "—"
        ) : (
          <>
            <CountValue value={value} digits={digits} />
            {suffix}
          </>
        )}
      </div>
      {detail && <div className="mt-1 text-[10px] text-muted-foreground">{detail}</div>}
    </div>
  );
}

function Darkroom() {
  const [prompt, setPrompt] = useState(SUGGESTIONS[1] ?? "");
  const [requestedTier, setRequestedTier] = useState<TierKey | "adaptive">("adaptive");
  const [device, setDevice] = useState<DeviceKey>("auto");
  const [memory, setMemory] = useState(6);
  const [tiers, setTiers] = useState<Record<TierKey, TierSpec>>(DEFAULT_TIERS);
  const [post, setPost] = useState<PostSettings>(DEFAULT_POST);
  const [autoRetry, setAutoRetry] = useState(true);
  const [connectionUrl, setConnectionUrl] = useState("");
  const [urlDraft, setUrlDraft] = useState("");
  const [connection, setConnection] = useState<ConnectionState>("demo");
  const [healthDevice, setHealthDevice] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [jobState, setJobState] = useState<"idle" | "loading" | "queued" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [stageStates, setStageStates] = useState<Record<StageKey, StageTiming>>(
    () =>
      Object.fromEntries(STAGES.map((stage) => [stage, { state: "wait" }])) as Record<
        StageKey,
        StageTiming
      >,
  );
  const [step, setStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(4);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<ResultState | null>(null);
  const [checks, setChecks] = useState<Check[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [compareValue, setCompareValue] = useState(50);
  const [promptChecking, setPromptChecking] = useState(false);
  const [fallbackMessage, setFallbackMessage] = useState("");
  const [allowanceOverride, setAllowanceOverride] = useState<number | null>(null);
  const [raceState, setRaceState] = useState<"idle" | "running" | "done">("idle");
  const [raceTimes, setRaceTimes] = useState<{ current: number; baseline: number } | null>(null);
  const [raceMem, setRaceMem] = useState<{ current: number; baseline: number } | null>(null);
  const [racePreview, setRacePreview] = useState("");
  const [baselinePreview, setBaselinePreview] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [showRails, setShowRails] = useState(true);
  const [statusMessage, setStatusMessage] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const inactivityRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const seedRef = useRef(0);
  const [mounted, setMounted] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const beginRef = useRef<(p: string, seed?: number, forceDemo?: boolean) => void>(() => {});
  const retryRef = useRef(false);
  const jobStartedRef = useRef(0);
  const historyIdRef = useRef(0);

  useEffect(() => {
    seedRef.current = Math.floor(Math.random() * 100000);
    setMounted(true);
    const localMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    if (typeof localMemory === "number") setMemory(localMemory);
    // ?server=wss://host/ws in the link connects any device (judges' phones) without settings,
    // and is remembered for later visits.
    const fromLink = new URLSearchParams(window.location.search).get("server")?.trim();
    if (fromLink && /^(wss?|https?):\/\//i.test(fromLink))
      localStorage.setItem("darkroom-ws-url", fromLink);
    const stored = fromLink || localStorage.getItem("darkroom-ws-url") || DEFAULT_URL;
    if (stored) {
      setConnectionUrl(stored);
      setUrlDraft(stored);
      setConnection("connecting");
    }
  }, []);

  const resolvedDevice = resolveDevice(device, memory);
  const allowance = allowanceOverride ?? DEVICE_ALLOWANCE[resolvedDevice];
  const tierDecision = resolveTier(requestedTier, allowance, tiers);
  const activeTier = tierDecision.tier;
  const activeSpec = tiers[activeTier];
  const backendBase = connectionUrl ? getBackendBase(connectionUrl) : "";
  const modeLabel =
    connection === "live"
      ? `Live · ${healthDevice || DEVICE_LABEL[resolvedDevice]}`
      : connection === "unreachable"
        ? "Unreachable"
        : "Demo mode — not model output";

  const checkServer = useCallback(async (url: string) => {
    if (!url.trim()) {
      setConnection("demo");
      setHealthDevice("");
      setAllowanceOverride(null);
      return;
    }
    setConnection("connecting");
    const base = getBackendBase(url.trim());
    try {
      const [healthResponse, tiersResponse] = await Promise.all([
        fetch(`${base}/health`),
        fetch(`${base}/tiers`),
      ]);
      if (!healthResponse.ok || !tiersResponse.ok)
        throw new Error(`Server returned ${healthResponse.status}/${tiersResponse.status}`);
      const health = (await healthResponse.json()) as {
        ok?: boolean;
        device?: string;
        tiers?: string[];
        mock?: boolean;
      };
      const tierPayload = (await tiersResponse.json()) as unknown;
      if (health.ok === false)
        throw new Error("The image server is still loading models. Try again in a minute.");
      setTiers(normalizeTiers(tierPayload));
      setHealthDevice(health.device ?? "");
      setConnection("live");
      setStatusMessage(
        health.mock ? "Connected to image server · mock pipeline" : "Connected to image server",
      );
      setAllowanceOverride(null);
    } catch (error) {
      setConnection("unreachable");
      setAllowanceOverride(null);
      setStatusMessage(
        error instanceof Error ? error.message : "Could not reach the image server.",
      );
    }
  }, []);

  useEffect(() => {
    if (!connectionUrl) return;
    void checkServer(connectionUrl);
  }, [checkServer, connectionUrl]);

  useEffect(
    () => () => {
      socketRef.current?.close();
      if (inactivityRef.current) clearTimeout(inactivityRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    },
    [],
  );

  const resetPipeline = useCallback(() => {
    setStageStates(
      Object.fromEntries(STAGES.map((stage) => [stage, { state: "wait" }])) as Record<
        StageKey,
        StageTiming
      >,
    );
    setStep(0);
    setTotalSteps(activeSpec.steps);
    setPreview("");
    setResult(null);
    setChecks([]);
    setFallbackMessage("");
    setElapsed(0);
  }, [activeSpec.steps]);

  const startClock = useCallback(() => {
    jobStartedRef.current = performance.now();
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(
      () => setElapsed((performance.now() - jobStartedRef.current) / 1000),
      60,
    );
  }, []);

  const finishClock = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    setElapsed((performance.now() - jobStartedRef.current) / 1000);
  }, []);

  const recordResult = useCallback((next: ResultState, sourcePrompt: string) => {
    setResult(next);
    setChecks(next.checks);
    const image = next.processed || next.raw;
    setPreview(image);
    setHistory((current) =>
      [{ ...next, id: ++historyIdRef.current, prompt: sourcePrompt, image }, ...current].slice(
        0,
        12,
      ),
    );
  }, []);

  const parseMetrics = useCallback((input: unknown): Metrics => {
    const m = input && typeof input === "object" ? (input as Partial<Metrics>) : {};
    return {
      clip: typeof m.clip === "number" ? m.clip : null,
      peak_mb: typeof m.peak_mb === "number" ? m.peak_mb : null,
      model_mb: typeof m.model_mb === "number" ? m.model_mb : null,
      encode_ms: Number(m.encode_ms) || 0,
      denoise_ms: Number(m.denoise_ms) || 0,
      decode_ms: Number(m.decode_ms) || 0,
      post_ms: Number(m.post_ms) || 0,
      qa_ms: Number(m.qa_ms) || 0,
      verify_ms: Number(m.verify_ms) || 0,
      total_ms: Number(m.total_ms) || 0,
    };
  }, []);

  const setStage = useCallback((stage: StageKey, state: StageState, ms?: number) => {
    setStageStates((current) => ({ ...current, [stage]: { state, ms } }));
  }, []);

  const handleBackendMessage = useCallback(
    (message: BackendMessage, currentTier: TierKey, sourcePrompt: string) => {
      if (
        message.type === "stage" &&
        typeof message.stage === "string" &&
        STAGES.includes(message.stage as StageKey)
      ) {
        const stage = message.stage as StageKey;
        if (message.state === "run") setStage(stage, "run");
        if (message.state === "ok")
          setStage(stage, "ok", typeof message.ms === "number" ? message.ms : undefined);
      } else if (message.type === "step") {
        setStep(Number(message.step) || 0);
        setTotalSteps(Number(message.total) || activeSpec.steps);
        if (typeof message.preview === "string") setPreview(message.preview);
        setJobState("loading");
      } else if (message.type === "queued") {
        setJobState("queued");
        setStatusMessage(typeof message.message === "string" ? message.message : "Waiting for GPU");
      } else if (message.type === "fallback") {
        const from = typeof message.from === "string" ? message.from : "requested tier";
        setFallbackMessage(
          typeof message.message === "string"
            ? message.message
            : `Server selected a safer tier instead of ${from}.`,
        );
      } else if (message.type === "result") {
        const data = message as BackendMessage & {
          image?: string;
          processed?: string;
          metrics?: unknown;
          qa?: Record<string, unknown>;
          detections?: Detection[];
          checks?: Check[];
          tier_used?: string;
        };
        const actualTier = (
          ["baseline", "quality", "balanced", "lite"].includes(data.tier_used ?? "")
            ? data.tier_used
            : currentTier
        ) as TierKey;
        const payload: ResultState = {
          raw: data.image ?? "",
          processed: data.processed ?? data.image ?? "",
          metrics: parseMetrics(data.metrics),
          qa: data.qa ?? {},
          detections: data.detections ?? [],
          checks: data.checks ?? [],
          tier: actualTier,
        };
        STAGES.forEach((stage) =>
          setStage(stage, "ok", (payload.metrics[STAGE_TIMING_KEYS[stage]] as number) || undefined),
        );
        recordResult(payload, sourcePrompt);
        setJobState("idle");
        setPromptChecking(false);
        finishClock();
        if (
          autoRetry &&
          !retryRef.current &&
          payload.checks.some((check) => !check.ok && !check.muted)
        ) {
          retryRef.current = true;
          window.setTimeout(() => beginRef.current(sourcePrompt, seedRef.current + 1), 450);
        } else retryRef.current = false;
      } else if (message.type === "error") {
        const data = message as BackendMessage & { message?: string; fallback_tier?: string };
        setErrorMessage(data.message ?? "The image server reported an error.");
        if (data.fallback_tier)
          setFallbackMessage(`Server suggested ${data.fallback_tier} for another attempt.`);
        setJobState("error");
        setPromptChecking(false);
        finishClock();
      }
    },
    [activeSpec.steps, autoRetry, finishClock, parseMetrics, recordResult, setStage],
  );

  const runDemo = useCallback(
    (sourcePrompt: string, seed: number, tier: TierKey) => {
      setConnection((current) => (current === "unreachable" ? "unreachable" : "demo"));
      const spec = tiers[tier];
      const scene = makeDemoScene(sourcePrompt, spec.steps, seed, post);
      setTotalSteps(spec.steps);
      setJobState("loading");
      setStatusMessage("Composing a procedural scene");
      let index = 0;
      const stageOrder: StageKey[] = ["encode", "denoise", "decode", "post", "qa", "verify"];
      const stageDurations = [18, 22, 16, 14, 14, 16];
      const stageNext = (stageIndex: number) => {
        if (stageIndex >= stageOrder.length) {
          finishClock();
          const metrics: Metrics = {
            clip: spec.measured.clip,
            peak_mb: spec.measured.peakMB,
            model_mb: spec.measured.sizeMB,
            encode_ms: spec.measured.encMs,
            denoise_ms: spec.measured.gen_ms_mean - spec.measured.encMs - spec.measured.decMs,
            decode_ms: spec.measured.decMs,
            post_ms: 24,
            qa_ms: 21,
            verify_ms: 15,
            total_ms: spec.measured.gen_ms_mean,
          };
          const resultState: ResultState = {
            raw: scene.raw,
            processed: scene.processed,
            metrics,
            qa: {
              sharpness: 3538,
              clipped_pct: 0.05,
              colorfulness: 0.71,
              checkerboard: false,
              fixes: [],
            },
            detections: scene.detections,
            checks: scene.checks,
            tier,
          };
          recordResult(resultState, sourcePrompt);
          setJobState("idle");
          setPromptChecking(false);
          if (
            autoRetry &&
            !retryRef.current &&
            resultState.checks.some((check) => !check.ok && !check.muted)
          ) {
            retryRef.current = true;
            window.setTimeout(() => beginRef.current(sourcePrompt, seed + 1), 450);
          } else retryRef.current = false;
          return;
        }
        const stage = stageOrder[stageIndex] as StageKey;
        setStage(stage, "run");
        if (stage === "denoise") {
          const deliver = () => {
            if (index < scene.frames.length) {
              setStep(index + 1);
              setPreview(scene.frames[index] ?? "");
              index += 1;
              window.setTimeout(deliver, 360);
            } else
              window.setTimeout(() => {
                setStage(
                  stage,
                  "ok",
                  spec.measured.gen_ms_mean - spec.measured.encMs - spec.measured.decMs,
                );
                stageNext(stageIndex + 1);
              }, 220);
          };
          deliver();
        } else {
          window.setTimeout(() => {
            const ms =
              stage === "encode"
                ? spec.measured.encMs
                : stage === "decode"
                  ? spec.measured.decMs
                  : stageDurations[stageIndex];
            setStage(stage, "ok", ms);
            stageNext(stageIndex + 1);
          }, stageDurations[stageIndex]);
        }
      };
      stageNext(0);
    },
    [autoRetry, finishClock, post, recordResult, setStage, tiers],
  );

  const beginGeneration = useCallback(
    (sourcePrompt: string, seed = seedRef.current, forceDemo = false) => {
      if (!sourcePrompt.trim()) return;
      cancelRef.current?.();
      resetPipeline();
      setJobState("loading");
      setPromptChecking(true);
      setErrorMessage("");
      setElapsed(0);
      seedRef.current = seed;
      startClock();
      const selection = resolveTier(requestedTier, allowance, tiers);
      const selectedTier = selection.tier;
      setTotalSteps(tiers[selectedTier].steps);
      if (selection.downgraded && selection.requested) {
        setFallbackMessage(
          `${TIER_LABEL[selection.requested]} needs ${tiers[selection.requested].measured.peakMB.toLocaleString()} MB; this device allows ${allowance.toLocaleString()} MB. Switched to ${TIER_LABEL[selectedTier]}.`,
        );
      }
      if (!forceDemo && connectionUrl && connection === "live") {
        const job = runJob(
          connectionUrl,
          {
            prompt: sourcePrompt,
            seed,
            tier: selectedTier,
            device: { ram_gb: memory, allow_mb: allowance },
            preview: true,
            post: {
              contrast: post.contrast,
              wb: post.wb,
              sharp: post.sharp,
              upscale: post.upscale,
            },
          },
          (message) => handleBackendMessage(message, selectedTier, sourcePrompt),
        );
        cancelRef.current = job.cancel;
        job.promise.catch((error: Error) => {
          setErrorMessage(error.message);
          setJobState("error");
          setPromptChecking(false);
          finishClock();
        });
      } else runDemo(sourcePrompt, seed, selectedTier);
    },
    [
      allowance,
      connection,
      connectionUrl,
      finishClock,
      handleBackendMessage,
      memory,
      post,
      requestedTier,
      resetPipeline,
      runDemo,
      startClock,
      tiers,
    ],
  );
  beginRef.current = beginGeneration;

  const submit = useCallback(
    (event?: FormEvent) => {
      event?.preventDefault();
      // an empty box generates the first example instead of silently doing nothing
      const text = prompt.trim() || SUGGESTIONS[0] || "";
      if (!prompt.trim()) setPrompt(text);
      retryRef.current = false;
      beginGeneration(text);
    },
    [beginGeneration, prompt],
  );

  const handlePromptKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const applyHistory = (entry: HistoryEntry) => {
    setPrompt(entry.prompt);
    setResult(entry);
    setPreview(entry.image);
    setChecks(entry.checks);
    setJobState("idle");
    setStageStates(
      Object.fromEntries(
        STAGES.map((stage) => [
          stage,
          { state: "ok", ms: entry.metrics[STAGE_TIMING_KEYS[stage]] as number },
        ]),
      ) as Record<StageKey, StageTiming>,
    );
  };

  const closeServer = (url = "") => {
    socketRef.current?.close();
    setConnectionUrl(url);
    setUrlDraft(url);
    localStorage.setItem("darkroom-ws-url", url);
    setDialogOpen(false);
    if (!url) {
      setConnection("demo");
      setTiers(DEFAULT_TIERS);
      setAllowanceOverride(null);
      setHealthDevice("");
    } else void checkServer(url);
  };

  const [raceError, setRaceError] = useState("");
  const [raceClock, setRaceClock] = useState<{ current: number | null; baseline: number | null }>({
    current: null,
    baseline: null,
  });

  /** Live: run the current tier, then the FP32 baseline, on the real server (one GPU, so sequential).
   *  Each clock starts when the server starts that job and reports the server-measured generation time.
   *  Demo: replay the tier table timings so the comparison is still visible. */
  const runRace = async () => {
    if (raceState === "running") return;
    setRaceState("running");
    setRaceTimes(null);
    setRaceMem(null);
    setRacePreview("");
    setBaselinePreview("");
    setRaceError("");
    setRaceClock({ current: null, baseline: null });
    const racePrompt = prompt.trim() || SUGGESTIONS[0] || "";
    const seed = seedRef.current;
    const lanes: { key: "current" | "baseline"; tier: TierKey; setImg: (s: string) => void }[] = [
      { key: "current", tier: activeTier, setImg: setRacePreview },
      { key: "baseline", tier: "baseline", setImg: setBaselinePreview },
    ];
    const live = connection === "live" && !!connectionUrl;
    try {
      if (live && !(tiers.baseline as TierSpec & { available?: boolean }).available) {
        throw new Error(
          "The FP32 baseline is not loaded on this server (start it with DARKROOM_BASELINE=1).",
        );
      }
      for (const lane of lanes) {
        let t0: number | null = null;
        const tick = window.setInterval(() => {
          if (t0 !== null)
            setRaceClock((c) => ({ ...c, [lane.key]: (performance.now() - t0!) / 1000 }));
        }, 50);
        try {
          if (live) {
            const job = runJob(
              connectionUrl,
              {
                prompt: racePrompt,
                seed,
                tier: lane.tier,
                preview: true,
                device: { ram_gb: memory, allow_mb: allowance },
                post,
              },
              (m) => {
                if (t0 === null && (m.type === "stage" || m.type === "step"))
                  t0 = performance.now();
                if (m.type === "step" && typeof m.preview === "string") lane.setImg(m.preview);
              },
            );
            const res = await job.promise;
            const metrics = parseMetrics(res.metrics);
            const gen = (metrics.encode_ms + metrics.denoise_ms + metrics.decode_ms) / 1000;
            if (typeof res.processed === "string") lane.setImg(res.processed);
            setRaceTimes((old) => ({
              current: old?.current ?? 0,
              baseline: old?.baseline ?? 0,
              [lane.key]: gen,
            }));
            setRaceMem((old) => ({
              current: old?.current ?? 0,
              baseline: old?.baseline ?? 0,
              [lane.key]: metrics.peak_mb ?? tiers[lane.tier].measured.peakMB,
            }));
          } else {
            const spec = tiers[lane.tier];
            const scene = makeDemoScene(racePrompt, Math.min(spec.steps, 12), seed, post);
            t0 = performance.now();
            const ms = Math.min(spec.measured.gen_ms_mean, 6000);
            for (let i = 0; i < scene.frames.length; i++) {
              lane.setImg(scene.frames[i] ?? "");
              await new Promise((r) => setTimeout(r, ms / scene.frames.length));
            }
            lane.setImg(scene.processed);
            setRaceTimes((old) => ({
              current: old?.current ?? 0,
              baseline: old?.baseline ?? 0,
              [lane.key]: spec.measured.gen_ms_mean / 1000,
            }));
            setRaceMem((old) => ({
              current: old?.current ?? 0,
              baseline: old?.baseline ?? 0,
              [lane.key]: spec.measured.peakMB,
            }));
          }
        } finally {
          window.clearInterval(tick);
          setRaceClock((c) => ({ ...c, [lane.key]: null }));
        }
      }
      setRaceState("done");
    } catch (error) {
      setRaceError(error instanceof Error ? error.message : "The race failed.");
      setRaceState("idle");
    }
  };

  const chartData = useMemo(
    () =>
      Object.entries(tiers).map(([key, spec]) => ({
        tier: key,
        latency: Math.max(10, Math.round(spec.measured.gen_ms_mean)),
        clip: spec.measured.clip,
        memory: spec.measured.peakMB,
        fill:
          key === activeTier
            ? "var(--darkroom-coral)"
            : key === "baseline"
              ? "var(--darkroom-dim)"
              : "var(--darkroom-teal)",
      })),
    [activeTier, tiers],
  );
  const stageTotal = STAGES.reduce((sum, stage) => sum + (stageStates[stage].ms ?? 0), 0);
  const latency = result
    ? (result.metrics.total_ms || elapsed * 1000) / 1000
    : tiers[activeTier].measured.gen_ms_mean / 1000;
  const resultGenMs = result
    ? result.metrics.encode_ms + result.metrics.denoise_ms + result.metrics.decode_ms
    : 0;
  const baselineFast =
    tiers.baseline.measured.gen_ms_mean /
    Math.max(1, resultGenMs || tiers[activeTier].measured.gen_ms_mean);
  const resultChecks = checks.filter((check) => !check.muted);
  const failedChecks = resultChecks.filter((check) => !check.ok).length;
  const qaValue = (key: string): string =>
    typeof result?.qa[key] === "number"
      ? String(result.qa[key])
      : typeof result?.qa[key] === "boolean"
        ? result.qa[key]
          ? "Yes"
          : "No"
        : "—";

  const [view, setView] = useState<"studio" | "bench" | "how">("studio");
  const prevJob = useRef(jobState);
  useEffect(() => {
    const was = prevJob.current;
    prevJob.current = jobState;
    if ((was === "loading" || was === "queued") && jobState === "idle" && result) {
      const m = result.metrics;
      const gen = (m.encode_ms + m.denoise_ms + m.decode_ms) / 1000;
      const speed = tiers.baseline.measured.gen_ms_mean / 1000 / Math.max(0.01, gen);
      const failed = result.checks.filter((c) => !c.ok).length;
      toast(`Developed in ${gen.toFixed(2)} s`, {
        description: `${TIER_LABEL[result.tier]} tier · ${speed.toFixed(0)}× faster than the baseline${
          result.checks.length
            ? ` · ${result.checks.length - failed}/${result.checks.length} checks passed`
            : ""
        }`,
        className:
          "!bg-darkroom-panel !border-darkroom-panel-edge !text-foreground !backdrop-blur-xl !rounded-2xl",
      });
    }
  }, [jobState, result, tiers]);
  const busy = jobState === "loading" || jobState === "queued";
  const isLive = connection === "live";
  const shownImage = result ? result.processed || result.raw : preview;
  const tierOrder: TierKey[] = ["baseline", "quality", "balanced", "lite"];
  const base = tiers.baseline.measured;
  const balanced = tiers.balanced.measured;
  const lite = tiers.lite;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-darkroom-canvas text-foreground selection:bg-darkroom-violet/30">
      <div className="darkroom-aurora" aria-hidden />
      <LatentField active={busy} />
      <Toaster position="bottom-center" theme="dark" />
      <div
        className="pointer-events-none fixed inset-0 z-[1] bg-[radial-gradient(ellipse_at_50%_30%,transparent_10%,var(--darkroom-canvas)_95%)] opacity-90"
        aria-hidden
      />

      <div className="relative z-10 mx-auto max-w-[1400px] px-4 pb-16 sm:px-6 lg:px-10">
        {/* ------------------------------------------------------------ header */}
        <header className="flex flex-wrap items-center justify-between gap-4 py-5">
          <div className="flex items-center gap-3">
            <div className="darkroom-spin-border grid h-11 w-11 place-items-center rounded-2xl p-[2px] shadow-[0_8px_30px_-8px_var(--darkroom-violet)]">
              <div className="grid h-full w-full place-items-center rounded-[14px] bg-darkroom-canvas">
                <Moon className="h-5 w-5 text-darkroom-amber" />
              </div>
            </div>
            <div>
              <div className="font-display text-2xl font-semibold leading-none tracking-tight">
                Darkroom<span className="text-darkroom-coral">.</span>
              </div>
              <div className="mt-1 text-sm text-muted-foreground">
                Text-to-image, sized for your phone
              </div>
            </div>
          </div>

          <nav
            className="order-3 flex w-full gap-1 rounded-2xl border border-darkroom-panel-edge bg-darkroom-panel p-1 backdrop-blur-xl sm:order-none sm:w-auto"
            aria-label="Views"
          >
            {(
              [
                ["studio", "Studio", Sparkles],
                ["bench", "Benchmarks", Gauge],
                ["how", "How it works", Layers3],
              ] as const
            ).map(([key, label, Icon]) => (
              <button
                key={key}
                type="button"
                onClick={() => setView(key)}
                aria-current={view === key ? "page" : undefined}
                className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition sm:flex-none ${
                  view === key ? "darkroom-active" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span className="whitespace-nowrap sm:hidden">
                  {label === "How it works" ? "How" : label === "Benchmarks" ? "Bench" : label}
                </span>
                <span className="hidden whitespace-nowrap sm:inline">{label}</span>
              </button>
            ))}
          </nav>

          <button
            type="button"
            onClick={() => {
              setUrlDraft(connectionUrl);
              setDialogOpen(true);
            }}
            className="flex items-center gap-2.5 rounded-full border border-darkroom-panel-edge bg-darkroom-panel px-4 py-2.5 text-sm backdrop-blur-xl hover:border-darkroom-violet/50"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isLive
                  ? "bg-darkroom-success shadow-[0_0_10px_var(--darkroom-success)]"
                  : connection === "unreachable"
                    ? "bg-darkroom-coral"
                    : connection === "connecting"
                      ? "darkroom-gradient darkroom-pulse"
                      : "bg-darkroom-amber"
              }`}
            />
            <span className="font-medium">
              {isLive
                ? `Live model${healthDevice ? ` · ${healthDevice}` : ""}`
                : connection === "unreachable"
                  ? "Server unreachable"
                  : connection === "connecting"
                    ? "Connecting…"
                    : "Demo mode"}
            </span>
            <Settings2 className="h-4 w-4 text-muted-foreground" />
          </button>
        </header>

        {!isLive && (
          <div className="mb-6 flex flex-col gap-3 rounded-2xl border border-darkroom-amber/25 bg-darkroom-amber/[0.07] px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <Wifi className="mt-0.5 h-5 w-5 shrink-0 text-darkroom-amber" />
              <p className="text-sm leading-relaxed text-foreground/90">
                <span className="font-semibold text-darkroom-amber">
                  {connection === "unreachable"
                    ? "The model server isn't reachable. "
                    : "Demo mode. "}
                </span>
                Images here are procedural previews, not model output. Timings and scores come from
                our T4 evaluation. Connect the model server to generate for real.
              </p>
            </div>
            <Button
              className="darkroom-active h-10 shrink-0 rounded-xl px-5 text-sm"
              onClick={() => {
                setUrlDraft(connectionUrl);
                setDialogOpen(true);
              }}
            >
              <Link2 className="h-4 w-4" />
              Connect server
            </Button>
          </div>
        )}

        <AnimatePresence mode="wait">
          <motion.main
            key={view}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22 }}
          >
            {/* ========================================================== STUDIO */}
            {view === "studio" && (
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_420px]">
                <div className="min-w-0 space-y-5">
                  {/* stage */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
                    className={`mx-auto w-full max-w-[720px] rounded-[26px] p-[2px] transition-shadow duration-700 ${
                      busy
                        ? "darkroom-spin-border shadow-[0_0_80px_-10px_var(--darkroom-violet)]"
                        : "bg-darkroom-panel-edge shadow-[0_40px_120px_-50px_var(--darkroom-violet)]"
                    }`}
                  >
                    <div className="relative aspect-square w-full overflow-hidden rounded-3xl bg-darkroom-canvas">
                      {shownImage ? (
                        <>
                          <motion.img
                            key={result ? `r-${result.processed.length}-${result.tier}` : "preview"}
                            initial={
                              result ? { opacity: 0.4, scale: 1.04, filter: "blur(10px)" } : false
                            }
                            animate={{
                              opacity: 1,
                              scale: 1,
                              filter: busy
                                ? `blur(${Math.max(0, (1 - step / Math.max(1, totalSteps)) * 14)}px) saturate(${0.6 + (step / Math.max(1, totalSteps)) * 0.4})`
                                : "blur(0px) saturate(1)",
                            }}
                            transition={{ duration: 0.7, ease: [0.2, 0.8, 0.2, 1] }}
                            className="darkroom-stage-image absolute inset-0 h-full w-full object-cover"
                            src={result?.raw && result.processed ? result.raw : shownImage}
                            alt={prompt ? `Generated scene: ${prompt}` : "Generated scene"}
                          />
                          {result?.raw && result.processed && (
                            <img
                              className="darkroom-stage-image absolute inset-0 h-full w-full object-cover"
                              src={result.processed}
                              alt=""
                              style={{ clipPath: `inset(0 0 0 ${compareValue}%)` }}
                            />
                          )}
                          {result?.raw && (
                            <>
                              <div
                                className="pointer-events-none absolute inset-y-0 z-[5] w-0.5 bg-white/90 shadow-[0_0_14px_rgba(0,0,0,.7)]"
                                style={{ left: `${compareValue}%` }}
                              >
                                <span className="absolute left-1/2 top-1/2 grid h-10 w-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/50 bg-black/60 text-white backdrop-blur">
                                  <ArrowLeftRight className="h-5 w-5" />
                                </span>
                              </div>
                              <input
                                className="absolute inset-0 z-[6] h-full w-full cursor-ew-resize opacity-0"
                                type="range"
                                min="0"
                                max="100"
                                value={compareValue}
                                onChange={(e) => setCompareValue(Number(e.target.value))}
                                aria-label="Compare raw decode and post-processed image"
                              />
                              <span className="pointer-events-none absolute bottom-4 left-4 z-[7] rounded-full bg-black/60 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">
                                Raw decode
                              </span>
                              <span className="pointer-events-none absolute bottom-4 right-4 z-[7] rounded-full bg-black/60 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">
                                Post-processed
                              </span>
                            </>
                          )}
                          {showBoxes &&
                            result?.detections.map((d, i) => (
                              <div
                                key={`${d.label}-${i}`}
                                className="pointer-events-none absolute z-[8] rounded-md border-2"
                                style={{
                                  borderColor: d.ok
                                    ? "var(--darkroom-teal)"
                                    : "var(--darkroom-coral)",
                                  borderStyle: d.ok ? "solid" : "dashed",
                                  left: `${d.box[0] * 100}%`,
                                  top: `${d.box[1] * 100}%`,
                                  width: `${(d.box[2] - d.box[0]) * 100}%`,
                                  height: `${(d.box[3] - d.box[1]) * 100}%`,
                                }}
                              >
                                <span
                                  className="absolute -top-7 left-0 whitespace-nowrap rounded-md px-2 py-1 text-xs font-semibold text-black"
                                  style={{
                                    background: d.ok
                                      ? "var(--darkroom-teal)"
                                      : "var(--darkroom-coral)",
                                  }}
                                >
                                  {d.label} {Math.round(d.score * 100)}%
                                </span>
                              </div>
                            ))}
                        </>
                      ) : (
                        <div className="absolute inset-0 grid place-items-center p-8 text-center">
                          <div className="darkroom-latent-idle absolute inset-[-10%]" aria-hidden />
                          <div className="darkroom-grain-live opacity-20" aria-hidden />
                          <div className="relative">
                            <div className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-2xl border border-darkroom-panel-edge bg-darkroom-panel">
                              <Sparkles className="h-7 w-7 text-darkroom-violet" />
                            </div>
                            <p className="font-display text-2xl font-semibold sm:text-3xl">
                              {busy ? "Forming from noise…" : "Describe a scene."}
                            </p>
                            <p className="mx-auto mt-3 max-w-sm text-base text-muted-foreground">
                              {busy
                                ? "The first frames arrive in a moment."
                                : "It will form out of noise in a few steps. Try one of the examples below."}
                            </p>
                          </div>
                        </div>
                      )}
                      {busy && (
                        <div
                          className="pointer-events-none absolute inset-0 z-[15] overflow-hidden"
                          aria-hidden
                        >
                          <div className="darkroom-grain-live" />
                          <div className="darkroom-scanline" />
                        </div>
                      )}
                      {busy && (
                        <div className="absolute right-4 top-4 z-20 flex items-center gap-3 rounded-full border border-white/10 bg-black/60 py-1.5 pl-1.5 pr-4 backdrop-blur-xl">
                          <div className="relative grid h-11 w-11 place-items-center rounded-full p-[3px]">
                            <div
                              className="absolute inset-0 rounded-full"
                              style={{
                                background: `conic-gradient(var(--darkroom-violet) ${
                                  (step / Math.max(1, totalSteps)) * 360
                                }deg, rgba(255,255,255,.12) 0deg)`,
                              }}
                            />
                            <div className="relative grid h-full w-full place-items-center rounded-full bg-black text-xs font-semibold tabular-nums text-white">
                              {step}/{totalSteps}
                            </div>
                          </div>
                          <span className="text-sm font-medium text-white">
                            {jobState === "queued" ? "Waiting for GPU" : "Denoising"}
                          </span>
                        </div>
                      )}
                      {result && !busy && (
                        <div className="absolute left-4 top-4 z-[9] rounded-full bg-black/60 px-3.5 py-2 text-sm font-medium tabular-nums text-white backdrop-blur">
                          {TIER_LABEL[result.tier]} · {tiers[result.tier].steps} steps ·{" "}
                          {latency.toFixed(2)} s
                        </div>
                      )}
                    </div>
                  </motion.div>

                  {jobState === "error" && (
                    <div className="mx-auto flex max-w-[720px] flex-col gap-3 rounded-2xl border border-darkroom-coral/30 bg-darkroom-coral/10 p-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex gap-3">
                        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-darkroom-coral" />
                        <div>
                          <div className="font-medium">Generation stopped</div>
                          <div className="mt-1 text-sm text-muted-foreground">{errorMessage}</div>
                        </div>
                      </div>
                      <div className="flex shrink-0 gap-2">
                        <Button
                          variant="outline"
                          className="border-darkroom-panel-edge"
                          onClick={() => {
                            setRequestedTier("lite");
                            window.setTimeout(() => beginRef.current(prompt, seedRef.current), 0);
                          }}
                        >
                          <RefreshCw className="h-4 w-4" />
                          Retry on Lite
                        </Button>
                        <Button
                          variant="secondary"
                          onClick={() => {
                            closeServer("");
                            beginGeneration(prompt, seedRef.current, true);
                          }}
                        >
                          Use demo mode
                        </Button>
                      </div>
                    </div>
                  )}
                  {fallbackMessage && (
                    <div className="mx-auto flex max-w-[720px] items-start gap-3 rounded-2xl border border-darkroom-warning/30 bg-darkroom-warning/10 px-4 py-3 text-sm text-darkroom-warning">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                      <span>{fallbackMessage}</span>
                    </div>
                  )}

                  {/* prompt composer */}
                  <form
                    className="darkroom-glass mx-auto max-w-[720px] rounded-3xl p-4 sm:p-5"
                    onSubmit={submit}
                  >
                    <label htmlFor="prompt" className="text-sm font-medium text-muted-foreground">
                      Your prompt
                    </label>
                    <div className="mt-2 flex flex-col gap-3 sm:flex-row sm:items-end">
                      <Textarea
                        id="prompt"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={handlePromptKey}
                        placeholder="a red cube to the left of a blue sphere on a wooden table at sunset"
                        className="min-h-[88px] flex-1 resize-none rounded-2xl border-darkroom-panel-edge bg-darkroom-canvas/60 px-4 py-3 text-base leading-relaxed placeholder:text-muted-foreground/60 focus-visible:ring-darkroom-violet/50"
                      />
                      <Button
                        type="submit"
                        disabled={busy}
                        className="darkroom-active darkroom-shimmer h-14 shrink-0 gap-2 rounded-2xl px-7 text-base font-semibold transition-transform hover:scale-[1.03] active:scale-95 sm:w-auto"
                      >
                        {busy ? (
                          <LoaderCircle className="h-5 w-5 animate-spin" />
                        ) : (
                          <Sparkles className="h-5 w-5" />
                        )}
                        {busy ? "Generating" : "Generate"}
                      </Button>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {SUGGESTIONS.map((s) => (
                        <button
                          type="button"
                          key={s}
                          onClick={() => setPrompt(s)}
                          className="rounded-full border border-darkroom-panel-edge bg-darkroom-canvas/40 px-3.5 py-1.5 text-left text-sm text-muted-foreground transition hover:border-darkroom-violet/50 hover:text-foreground"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-darkroom-panel-edge pt-3 text-sm text-muted-foreground">
                      <button
                        type="button"
                        onClick={() => setShowBoxes((v) => !v)}
                        className="flex items-center gap-2 hover:text-foreground"
                      >
                        <Crosshair className="h-4 w-4 text-darkroom-teal" />
                        {showBoxes ? "Hide detection boxes" : "Show detection boxes"}
                      </button>
                      <span className="tabular-nums">
                        {mounted ? `Seed ${seedRef.current}` : ""} · Enter to generate
                      </span>
                    </div>
                  </form>

                  {/* metrics */}
                  <section aria-label="This image" className="mx-auto max-w-[720px]">
                    <h2 className="mb-3 font-display text-xl font-semibold">
                      {result ? "This image" : `${TIER_LABEL[activeTier]} tier, measured on a T4`}
                    </h2>
                    <motion.div
                      className="grid grid-cols-2 gap-3 sm:grid-cols-3"
                      initial="hidden"
                      animate="show"
                      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.06 } } }}
                    >
                      <BigStat
                        label="Generation time"
                        value={
                          result
                            ? `${(resultGenMs / 1000).toFixed(2)} s`
                            : `${(activeSpec.measured.gen_ms_mean / 1000).toFixed(2)} s`
                        }
                        note={`${baselineFast.toFixed(0)}× faster than baseline`}
                        accent
                      />
                      <BigStat
                        label="CLIP similarity"
                        value={
                          (result ? result.metrics.clip : activeSpec.measured.clip) !== null
                            ? (result
                                ? (result.metrics.clip ?? 0)
                                : activeSpec.measured.clip
                              ).toFixed(3)
                            : "—"
                        }
                        note={`baseline ${base.clip.toFixed(3)}`}
                      />
                      <BigStat
                        label="Model size"
                        value={`${((result?.metrics.model_mb ?? activeSpec.measured.sizeMB) / 1000).toFixed(2)} GB`}
                        note={`baseline ${(base.sizeMB / 1000).toFixed(2)} GB`}
                      />
                      <BigStat
                        label="Peak memory"
                        value={
                          result?.metrics.peak_mb
                            ? `${(result.metrics.peak_mb / 1000).toFixed(2)} GB`
                            : result
                              ? "—"
                              : `${(activeSpec.measured.peakMB / 1000).toFixed(1)} GB`
                        }
                        note={`device allows ${(allowance / 1000).toFixed(1)} GB`}
                      />
                      <BigStat
                        label="Sharpness"
                        value={
                          result && typeof result.qa["sharpness"] === "number"
                            ? Math.round(result.qa["sharpness"] as number).toLocaleString()
                            : "—"
                        }
                        note="Laplacian variance"
                      />
                      <BigStat
                        label="Clipped pixels"
                        value={
                          result && typeof result.qa["clipped_pct"] === "number"
                            ? `${(result.qa["clipped_pct"] as number).toFixed(2)}%`
                            : "—"
                        }
                        note="blown highlights + crushed shadows"
                      />
                    </motion.div>
                  </section>

                  {/* checks */}
                  <section
                    aria-label="Prompt verification"
                    className="darkroom-glass mx-auto max-w-[720px] rounded-3xl p-5"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h2 className="font-display text-xl font-semibold">
                          Does it match the prompt?
                        </h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Objects, colours and positions checked by an open-vocabulary detector.
                        </p>
                      </div>
                      <label className="flex items-center gap-3 text-sm text-muted-foreground">
                        Retry on failure
                        <Switch
                          checked={autoRetry}
                          onCheckedChange={setAutoRetry}
                          aria-label="Retry automatically when a check fails"
                        />
                      </label>
                    </div>
                    {checks.length ? (
                      <ul className="mt-4 space-y-2.5">
                        <li
                          className={`text-sm font-medium ${failedChecks ? "text-darkroom-coral" : "text-darkroom-success"}`}
                        >
                          {resultChecks.length - failedChecks} of {resultChecks.length} checks
                          passed
                        </li>
                        {checks.map((c, i) => (
                          <li key={`${c.text}-${i}`} className="flex items-center gap-3 text-base">
                            <span
                              className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${
                                c.ok
                                  ? "bg-darkroom-success/15 text-darkroom-success"
                                  : "bg-darkroom-coral/15 text-darkroom-coral"
                              }`}
                            >
                              {c.ok ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
                            </span>
                            <span className={c.ok ? "" : "text-darkroom-coral"}>{c.text}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-4 text-sm text-muted-foreground">
                        Checks appear when the image is ready.
                      </p>
                    )}
                  </section>

                  {history.length > 0 && (
                    <section aria-label="Recent images" className="mx-auto max-w-[720px]">
                      <h2 className="mb-3 font-display text-xl font-semibold">Recent</h2>
                      <div className="darkroom-scrollbar flex gap-3 overflow-x-auto pb-2">
                        {history.map((h) => (
                          <button
                            key={h.id}
                            type="button"
                            onClick={() => applyHistory(h)}
                            className="group w-28 shrink-0 text-left"
                          >
                            <img
                              src={h.image}
                              alt={h.prompt}
                              className="aspect-square w-28 rounded-xl border border-darkroom-panel-edge object-cover transition group-hover:border-darkroom-violet"
                            />
                            <span className="mt-1.5 block truncate text-xs text-muted-foreground">
                              {h.prompt}
                            </span>
                          </button>
                        ))}
                      </div>
                    </section>
                  )}
                </div>

                {/* sidebar */}
                <aside className="space-y-5 lg:sticky lg:top-5 lg:self-start">
                  <Panel
                    title="Compute budget"
                    icon={Cpu}
                    hint={requestedTier === "adaptive" ? "Adaptive" : "Manual"}
                  >
                    <FieldLabel>Device</FieldLabel>
                    <div className="grid grid-cols-2 gap-1.5">
                      {(["auto", "budget", "mid", "flagship"] as DeviceKey[]).map((key) => (
                        <button
                          key={key}
                          type="button"
                          onClick={() => setDevice(key)}
                          className={`rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                            device === key
                              ? "bg-secondary text-foreground ring-1 ring-darkroom-violet/50"
                              : "border border-darkroom-panel-edge text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {key === "auto"
                            ? `Auto (${DEVICE_LABEL[resolvedDevice].split(" ")[0]})`
                            : DEVICE_LABEL[key]}
                        </button>
                      ))}
                    </div>
                    <FieldLabel>Quality tier</FieldLabel>
                    <div className="grid grid-cols-2 gap-1.5">
                      {(["adaptive", "quality", "balanced", "lite"] as const).map((key) => (
                        <button
                          key={key}
                          type="button"
                          onClick={() => setRequestedTier(key)}
                          aria-pressed={requestedTier === key}
                          className={`rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                            requestedTier === key
                              ? "darkroom-active"
                              : "border border-darkroom-panel-edge text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {key === "adaptive" ? "Adaptive" : TIER_LABEL[key]}
                        </button>
                      ))}
                    </div>
                    <div className="mt-5">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Memory use</span>
                        <span className="font-medium tabular-nums">
                          {(activeSpec.measured.peakMB / 1000).toFixed(1)} /{" "}
                          {(allowance / 1000).toFixed(1)} GB
                        </span>
                      </div>
                      <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-secondary">
                        <div
                          className={`h-full rounded-full transition-all ${tierDecision.downgraded ? "bg-darkroom-warning" : "darkroom-gradient"}`}
                          style={{
                            width: `${Math.min(100, (activeSpec.measured.peakMB / allowance) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                    {tierDecision.downgraded && (
                      <div className="mt-4 flex gap-2.5 rounded-xl border border-darkroom-warning/30 bg-darkroom-warning/10 p-3 text-sm text-darkroom-warning">
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                        <span>
                          {TIER_LABEL[tierDecision.requested ?? activeTier]} needs more memory than
                          this device allows. Using {TIER_LABEL[activeTier]} instead.
                        </span>
                      </div>
                    )}
                    <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-2.5 border-t border-darkroom-panel-edge pt-4 text-sm">
                      <dt className="text-muted-foreground">Running</dt>
                      <dd className="text-right font-semibold">{TIER_LABEL[activeTier]}</dd>
                      <dt className="text-muted-foreground">Denoiser</dt>
                      <dd className="text-right">{activeSpec.unet}</dd>
                      <dt className="text-muted-foreground">Steps</dt>
                      <dd className="text-right tabular-nums">{activeSpec.steps}</dd>
                      <dt className="text-muted-foreground">Resolution</dt>
                      <dd className="text-right tabular-nums">{activeSpec.res} px</dd>
                      <dt className="text-muted-foreground">Precision</dt>
                      <dd className="text-right">{activeSpec.precision}</dd>
                      <dt className="text-muted-foreground">Decoder</dt>
                      <dd className="text-right">{activeSpec.decoder}</dd>
                    </dl>
                  </Panel>

                  <Panel
                    title="Live pipeline"
                    icon={Activity}
                    hint={busy ? "Running" : result ? "Done" : "Idle"}
                  >
                    <ol className="relative space-y-1">
                      <span
                        className="absolute bottom-3 left-[9px] top-3 w-px bg-darkroom-panel-edge"
                        aria-hidden
                      />
                      {STAGES.map((stage) => {
                        const s = stageStates[stage];
                        return (
                          <li
                            key={stage}
                            className="relative flex items-center justify-between gap-3 py-2"
                          >
                            <span className="flex items-center gap-3">
                              <span
                                className={`relative z-[1] h-[19px] w-[19px] rounded-full border-2 ${
                                  s.state === "run"
                                    ? "darkroom-gradient darkroom-ripple border-transparent"
                                    : s.state === "ok"
                                      ? "border-darkroom-teal bg-darkroom-teal"
                                      : "border-darkroom-panel-edge bg-darkroom-canvas"
                                }`}
                              />
                              <span
                                className={`text-base ${s.state === "wait" ? "text-muted-foreground" : ""}`}
                              >
                                {STAGE_LABEL[stage]}
                              </span>
                            </span>
                            <span className="text-sm tabular-nums text-muted-foreground">
                              {s.state === "run" ? (
                                <LoaderCircle className="h-4 w-4 animate-spin text-darkroom-violet" />
                              ) : s.ms !== undefined ? (
                                `${Math.round(s.ms)} ms`
                              ) : (
                                "—"
                              )}
                            </span>
                          </li>
                        );
                      })}
                    </ol>
                    <div className="mt-3 flex justify-between border-t border-darkroom-panel-edge pt-3 text-base">
                      <span className="text-muted-foreground">Total</span>
                      <span className="font-semibold tabular-nums">
                        {stageTotal > 0 ? `${Math.round(stageTotal)} ms` : "—"}
                      </span>
                    </div>
                  </Panel>

                  <Panel title="Classical post-processing" icon={WandSparkles} defaultOpen={false}>
                    {(
                      [
                        ["contrast", "Adaptive contrast", "CLAHE on luminance"],
                        ["wb", "White balance", "Gray-world correction"],
                        ["sharp", "Edge-aware sharpening", "Only along detected edges"],
                        ["upscale", "Upscale to 512 px", "Lanczos, for the 384 px Lite tier"],
                      ] as const
                    ).map(([key, label, sub]) => (
                      <label
                        key={key}
                        className="flex cursor-pointer items-center justify-between gap-3 border-b border-darkroom-panel-edge py-3 last:border-0"
                      >
                        <span>
                          <span className="block text-base">{label}</span>
                          <span className="block text-sm text-muted-foreground">{sub}</span>
                        </span>
                        <Switch
                          checked={post[key]}
                          onCheckedChange={(v) => setPost((old) => ({ ...old, [key]: v }))}
                          aria-label={label}
                        />
                      </label>
                    ))}
                  </Panel>
                </aside>
              </div>
            )}

            {/* ======================================================= BENCHMARKS */}
            {view === "bench" && (
              <div className="space-y-10">
                <section>
                  <motion.h1
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="font-display text-4xl font-semibold tracking-tight sm:text-6xl"
                  >
                    <span className="darkroom-gradient-text">
                      {Math.round(base.gen_ms_mean / Math.max(1, balanced.gen_ms_mean))}× faster.
                    </span>{" "}
                    {(base.sizeMB / Math.max(1, lite.measured.sizeMB)).toFixed(1)}× smaller.
                  </motion.h1>
                  <p className="mt-3 max-w-2xl text-lg text-muted-foreground">
                    The optimised tiers against the 50-step FP32 baseline, measured on a Tesla T4
                    with fixed seeds over COCO captions.
                  </p>
                  <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    {tierOrder.map((k, idx) => {
                      const t = tiers[k];
                      const m = t.measured;
                      const lo = Math.log10(200),
                        hi = Math.log10(Math.max(1000, base.gen_ms_mean));
                      const barPct = Math.max(
                        4,
                        ((Math.log10(Math.max(200, m.gen_ms_mean)) - lo) / (hi - lo)) * 100,
                      );
                      return (
                        <motion.div
                          key={k}
                          initial={{ opacity: 0, y: 24 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1 + idx * 0.08, duration: 0.5 }}
                          whileHover={{ y: -4 }}
                          className={`darkroom-glass rounded-3xl p-5 ${k === activeTier ? "ring-2 ring-darkroom-coral/70" : ""}`}
                        >
                          <div className="flex items-baseline justify-between">
                            <h3 className="font-display text-xl font-semibold">{t.name}</h3>
                            <span className="text-sm text-muted-foreground">
                              {t.steps} steps · {t.precision}
                            </span>
                          </div>
                          <div
                            className={`mt-4 font-display text-4xl font-semibold tabular-nums ${k === "baseline" ? "" : "text-darkroom-coral"}`}
                          >
                            {(m.gen_ms_mean / 1000).toFixed(m.gen_ms_mean < 10000 ? 2 : 1)} s
                          </div>
                          <div className="text-sm text-muted-foreground">per image</div>
                          <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${barPct}%` }}
                              transition={{
                                delay: 0.35 + idx * 0.1,
                                duration: 1.1,
                                ease: [0.2, 0.8, 0.2, 1],
                              }}
                              className={`h-full rounded-full ${k === "baseline" ? "bg-darkroom-dim" : "darkroom-gradient"}`}
                            />
                          </div>
                          <dl className="mt-4 grid grid-cols-2 gap-y-1.5 text-sm">
                            <dt className="text-muted-foreground">CLIP</dt>
                            <dd className="text-right tabular-nums">{m.clip.toFixed(3)}</dd>
                            <dt className="text-muted-foreground">Model size</dt>
                            <dd className="text-right tabular-nums">
                              {(m.sizeMB / 1000).toFixed(2)} GB
                            </dd>
                            <dt className="text-muted-foreground">Peak memory</dt>
                            <dd className="text-right tabular-nums">
                              {(m.peakMB / 1000).toFixed(2)} GB
                            </dd>
                            <dt className="text-muted-foreground">Decoder</dt>
                            <dd className="text-right">{t.decoder}</dd>
                            <dt className="text-muted-foreground">Resolution</dt>
                            <dd className="text-right tabular-nums">{t.res} px</dd>
                          </dl>
                        </motion.div>
                      );
                    })}
                  </div>
                </section>

                <section className="darkroom-glass rounded-3xl p-5 sm:p-7">
                  <div className="flex flex-wrap items-end justify-between gap-4">
                    <div>
                      <h2 className="font-display text-2xl font-semibold">The race</h2>
                      <p className="mt-1 text-base text-muted-foreground">
                        Same prompt and seed: the {TIER_LABEL[activeTier]} tier against full Stable
                        Diffusion 1.5.
                      </p>
                    </div>
                    <Button
                      className="darkroom-active h-12 rounded-2xl px-6 text-base"
                      onClick={runRace}
                      disabled={raceState === "running"}
                    >
                      {raceState === "running" ? (
                        <LoaderCircle className="h-5 w-5 animate-spin" />
                      ) : (
                        <Play className="h-5 w-5" />
                      )}
                      {raceState === "running"
                        ? "Racing…"
                        : raceState === "done"
                          ? "Race again"
                          : "Start the race"}
                    </Button>
                  </div>
                  <div className="mx-auto mt-6 grid max-w-4xl grid-cols-1 gap-5 md:grid-cols-2">
                    {(
                      [
                        {
                          key: "current",
                          label: TIER_LABEL[activeTier],
                          sub: `${activeSpec.steps} steps · ${activeSpec.precision}`,
                          img: racePreview,
                          time: raceClock.current ?? raceTimes?.current,
                          mem: raceMem?.current,
                          hot: true,
                        },
                        {
                          key: "base",
                          label: "Baseline",
                          sub: "SD 1.5 · FP32 · 50 steps",
                          img: baselinePreview,
                          time: raceClock.baseline ?? raceTimes?.baseline,
                          mem: raceMem?.baseline,
                          hot: false,
                        },
                      ] as const
                    ).map((lane) => (
                      <div
                        key={lane.key}
                        className="overflow-hidden rounded-2xl border border-darkroom-panel-edge bg-darkroom-canvas/50"
                      >
                        <div className="relative aspect-square">
                          {lane.img ? (
                            <img
                              src={lane.img}
                              alt={`${lane.label} output`}
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="grid h-full place-items-center text-muted-foreground">
                              <ImageIcon className="h-10 w-10 opacity-30" />
                            </div>
                          )}
                        </div>
                        <div className="p-5">
                          <div className="flex items-baseline justify-between">
                            <span className="font-display text-lg font-semibold">{lane.label}</span>
                            <span className="text-sm text-muted-foreground">{lane.sub}</span>
                          </div>
                          <div
                            className={`mt-3 font-display text-5xl font-semibold tabular-nums ${lane.hot ? "text-darkroom-coral" : ""}`}
                          >
                            {lane.time ? `${lane.time.toFixed(2)}s` : "—"}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  {raceState === "done" && raceTimes && (
                    <div className="mt-5 rounded-2xl border border-darkroom-teal/30 bg-darkroom-teal/10 p-4 text-center text-lg text-darkroom-teal">
                      <span className="font-semibold">
                        {(raceTimes.baseline / Math.max(0.01, raceTimes.current)).toFixed(1)}×
                        faster
                      </span>{" "}
                      on the same prompt and seed
                    </div>
                  )}
                  {raceError && <p className="mt-4 text-sm text-darkroom-coral">{raceError}</p>}
                  {!isLive && (
                    <p className="mt-4 text-sm text-muted-foreground">
                      Demo mode replays the measured T4 timings. Connect the server to race the real
                      models.
                    </p>
                  )}
                </section>

                <section className="darkroom-glass rounded-3xl p-5 sm:p-7">
                  <h2 className="font-display text-2xl font-semibold">Quality against cost</h2>
                  <p className="mt-1 text-base text-muted-foreground">
                    Each point is one tier. Up and to the left is better; bubble size is model size.
                  </p>
                  <div className="mt-6 h-[360px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                        <CartesianGrid stroke="var(--darkroom-panel-edge)" strokeDasharray="3 5" />
                        <XAxis
                          dataKey="latency"
                          type="number"
                          scale="log"
                          domain={[
                            Math.min(...chartData.map((d) => d.latency)) / 1.8,
                            Math.max(...chartData.map((d) => d.latency)) * 1.8,
                          ]}
                          ticks={[200, 500, 1000, 2000, 5000, 10000, 20000, 50000].filter(
                            (v) =>
                              v >= Math.min(...chartData.map((d) => d.latency)) / 1.8 &&
                              v <= Math.max(...chartData.map((d) => d.latency)) * 1.8,
                          )}
                          tickFormatter={(v: number) => `${v / 1000}s`}
                          tick={{ fill: "var(--darkroom-dim)", fontSize: 13 }}
                          label={{
                            value: "Generation time per image (log scale)",
                            position: "bottom",
                            offset: 10,
                            fill: "var(--darkroom-dim)",
                            fontSize: 13,
                          }}
                        />
                        <YAxis
                          dataKey="clip"
                          type="number"
                          domain={[
                            Math.floor((Math.min(...chartData.map((d) => d.clip)) - 0.006) * 100) /
                              100,
                            Math.ceil((Math.max(...chartData.map((d) => d.clip)) + 0.006) * 100) /
                              100,
                          ]}
                          tickFormatter={(v: number) => v.toFixed(2)}
                          tick={{ fill: "var(--darkroom-dim)", fontSize: 13 }}
                          width={48}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          content={({ active, payload }) => {
                            const p = payload?.[0]?.payload as
                              { tier?: TierKey; latency?: number; clip?: number } | undefined;
                            return active && p?.tier ? (
                              <div className="rounded-xl border border-darkroom-panel-edge bg-darkroom-canvas/95 px-3 py-2 text-sm shadow-xl">
                                <div className="font-semibold">{TIER_LABEL[p.tier]}</div>
                                <div className="text-muted-foreground">
                                  {((p.latency ?? 0) / 1000).toFixed(2)} s · CLIP{" "}
                                  {(p.clip ?? 0).toFixed(3)}
                                </div>
                              </div>
                            ) : null;
                          }}
                        />
                        <Scatter
                          data={chartData}
                          shape={(props: {
                            cx?: number;
                            cy?: number;
                            payload?: { tier?: string; fill?: string };
                          }) => {
                            const k = (props.payload?.tier ?? "lite") as TierKey;
                            const r = 8 + Math.sqrt(tiers[k].measured.sizeMB) / 6;
                            return (
                              <g>
                                <circle
                                  cx={props.cx}
                                  cy={props.cy}
                                  r={r}
                                  fill={props.payload?.fill}
                                  fillOpacity={0.85}
                                  stroke={k === activeTier ? "white" : "none"}
                                  strokeWidth={2}
                                />
                                <text
                                  x={(props.cx ?? 0) + r + 6}
                                  y={(props.cy ?? 0) + 5}
                                  fill="var(--foreground)"
                                  fontSize={14}
                                >
                                  {TIER_LABEL[k]}
                                </text>
                              </g>
                            );
                          }}
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="darkroom-glass overflow-x-auto rounded-3xl p-5 sm:p-7">
                  <h2 className="font-display text-2xl font-semibold">
                    What each optimisation buys
                  </h2>
                  <p className="mt-1 text-base text-muted-foreground">
                    Ablation on a T4: one factor changed at a time.
                  </p>
                  <table className="mt-5 w-full min-w-[560px] text-left text-base">
                    <thead className="text-sm text-muted-foreground">
                      <tr className="border-b border-darkroom-panel-edge">
                        <th className="py-3 font-medium">Change</th>
                        <th className="py-3 font-medium">Latency</th>
                        <th className="py-3 font-medium">CLIP</th>
                        <th className="py-3 font-medium">Takeaway</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ABLATION.map((r) => (
                        <tr
                          key={r[0]}
                          className="border-b border-darkroom-panel-edge/60 last:border-0"
                        >
                          <td className="py-3 pr-4 font-medium">{r[0]}</td>
                          <td className="py-3 pr-4 tabular-nums">{r[1]}</td>
                          <td className="py-3 pr-4 tabular-nums">{r[2]}</td>
                          <td className="py-3 text-muted-foreground">{r[3]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              </div>
            )}

            {/* ===================================================== HOW IT WORKS */}
            {view === "how" && (
              <div className="space-y-10">
                <section>
                  <motion.h1
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="font-display text-4xl font-semibold tracking-tight sm:text-6xl"
                  >
                    From prompt to <span className="darkroom-gradient-text">pixels</span> in six
                    stages
                  </motion.h1>
                  <p className="mt-3 max-w-2xl text-lg text-muted-foreground">
                    A Stable Diffusion 1.5 pipeline, distilled, quantised and wrapped in classical
                    computer vision.
                  </p>
                  <div className="darkroom-flow-line mt-8 hidden lg:block" aria-hidden>
                    <i />
                    <i />
                    <i />
                  </div>
                  <ol className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {PIPELINE.map((p, i) => (
                      <motion.li
                        key={p[0]}
                        initial={{ opacity: 0, y: 24 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.08, duration: 0.5 }}
                        whileHover={{ y: -5 }}
                        className="darkroom-glass group relative overflow-hidden rounded-3xl p-6"
                      >
                        <span
                          className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full opacity-0 blur-2xl transition-opacity duration-500 group-hover:opacity-60"
                          style={{
                            background: [
                              "var(--darkroom-violet)",
                              "var(--darkroom-coral)",
                              "var(--darkroom-amber)",
                              "var(--darkroom-teal)",
                              "var(--darkroom-violet)",
                              "var(--darkroom-coral)",
                            ][i],
                          }}
                          aria-hidden
                        />
                        <span className="font-display text-5xl font-semibold text-foreground/10">
                          0{i + 1}
                        </span>
                        <h3 className="mt-2 font-display text-xl font-semibold">{p[0]}</h3>
                        <p className="mt-2 text-base leading-relaxed text-muted-foreground">
                          {p[1]}
                        </p>
                      </motion.li>
                    ))}
                  </ol>
                </section>

                <section>
                  <h2 className="font-display text-2xl font-semibold">Five optimisations</h2>
                  <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {OPTIMISATIONS.map((o, i) => (
                      <motion.div
                        key={o[0]}
                        initial={{ opacity: 0, scale: 0.96 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ delay: i * 0.06, duration: 0.4 }}
                        whileHover={{ y: -4 }}
                        className="darkroom-glass rounded-3xl p-6"
                      >
                        <h3 className="font-display text-lg font-semibold">{o[0]}</h3>
                        <p className="mt-2 text-base leading-relaxed text-muted-foreground">
                          {o[1]}
                        </p>
                        <p className="darkroom-gradient-text mt-4 font-display text-2xl font-semibold">
                          {o[2]}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </section>

                <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="darkroom-glass rounded-3xl p-6">
                    <h2 className="font-display text-2xl font-semibold">Adapts to the device</h2>
                    <p className="mt-2 text-base leading-relaxed text-muted-foreground">
                      The app reads the device's memory and picks the best tier that fits. If you
                      force a tier that doesn't fit, it steps down and says why. If the GPU still
                      runs out of memory, the server retries on the next lighter tier.
                    </p>
                    <ul className="mt-4 space-y-2 text-base">
                      <li>
                        <span className="font-semibold">Budget phone</span>{" "}
                        <span className="text-muted-foreground">→ Lite</span>
                      </li>
                      <li>
                        <span className="font-semibold">Mid-range</span>{" "}
                        <span className="text-muted-foreground">→ Balanced</span>
                      </li>
                      <li>
                        <span className="font-semibold">Flagship</span>{" "}
                        <span className="text-muted-foreground">→ Quality</span>
                      </li>
                    </ul>
                  </div>
                  <div className="darkroom-glass rounded-3xl p-6">
                    <h2 className="font-display text-2xl font-semibold">Checks its own work</h2>
                    <p className="mt-2 text-base leading-relaxed text-muted-foreground">
                      Each prompt is parsed into objects, colours and positions ("red cube", "left
                      of"). OWL-ViT looks for every object and the boxes are checked against the
                      relation. A failed check triggers one retry with a new seed.
                    </p>
                    <p className="mt-4 text-base">
                      Weakest point, honestly: spatial relations. Relation accuracy is 0.19–0.31 for
                      every tier, including the full model.
                    </p>
                  </div>
                </section>
              </div>
            )}
          </motion.main>
        </AnimatePresence>

        <footer className="mt-16 border-t border-darkroom-panel-edge pt-6 text-sm text-muted-foreground">
          Images come from our own optimised diffusion pipeline, not a third-party image API.
          Computer Vision Hackathon 2026 · P3.
        </footer>
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-lg border-darkroom-panel-edge bg-darkroom-canvas text-foreground sm:rounded-3xl">
          <DialogHeader>
            <DialogTitle className="font-display text-2xl">Model server</DialogTitle>
            <DialogDescription className="text-base text-muted-foreground">
              Paste the WebSocket address of your model server. Leave it empty to stay in demo mode.
            </DialogDescription>
          </DialogHeader>
          <input
            value={urlDraft}
            onChange={(e) => setUrlDraft(e.target.value)}
            placeholder="wss://….trycloudflare.com/ws"
            className="h-12 w-full rounded-xl border border-darkroom-panel-edge bg-darkroom-panel px-4 text-base outline-none focus:border-darkroom-violet"
            aria-label="Server WebSocket URL"
          />
          <p className="text-sm text-muted-foreground">
            Status:{" "}
            {isLive
              ? `connected${healthDevice ? ` to ${healthDevice}` : ""}`
              : connection === "unreachable"
                ? `unreachable${statusMessage ? ` (${statusMessage})` : ""}`
                : connection === "connecting"
                  ? "checking…"
                  : "demo mode"}
          </p>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              className="border-darkroom-panel-edge"
              onClick={() => closeServer("")}
            >
              Use demo mode
            </Button>
            <Button className="darkroom-active" onClick={() => closeServer(urlDraft.trim())}>
              <Link2 className="h-4 w-4" />
              Connect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

const ABLATION: [string, string, string, string][] = [
  [
    "50 → 4 steps (LCM)",
    "31.8 s → 0.40 s",
    "0.308 → 0.301",
    "the biggest single win; also removes classifier-free guidance",
  ],
  ["4 → 8 steps", "575 → 957 ms", "0.297 → 0.293", "no measurable gain, so Balanced stays at 4"],
  ["Full VAE → TAESD", "577 → 400 ms", "0.297 → 0.296", "31% faster for free"],
  [
    "FP16 → INT8 UNet",
    "400 → 537 ms",
    "0.296 → 0.294",
    "half the weights; slower on a T4 without INT8 kernels",
  ],
  ["512 → 384 px", "416 → 351 ms", "0.299 → 0.293", "faster, but softer detail"],
  ["Post-processing on", "+ OpenCV chain", "unchanged", "clipped pixels 1.31% → 0.05%"],
];

const PIPELINE: [string, string][] = [
  [
    "Prompt encoding",
    "The CLIP text encoder turns the prompt into 77 embedding vectors that condition every denoising step.",
  ],
  [
    "Denoising",
    "A UNet with LCM-LoRA fused in removes noise from a 64×64 latent in 3–8 steps instead of 50, without a guidance pass.",
  ],
  [
    "Latent decode",
    "The tiny TAESD decoder (about 1M parameters) turns the latent into pixels, and also streams a preview of every step.",
  ],
  [
    "Classical CV",
    "Lanczos upscaling, gray-world white balance, CLAHE on luminance and sharpening only along detected edges.",
  ],
  [
    "Quality check",
    "Sharpness, clipped pixels and an FFT test for the 8-pixel checkerboard decoders can leave; problems are corrected.",
  ],
  [
    "Prompt check",
    "OWL-ViT finds each object named in the prompt and the boxes are tested against the requested positions.",
  ],
];

const OPTIMISATIONS: [string, string, string][] = [
  [
    "Fewer sampling steps",
    "Latent consistency distillation (LCM-LoRA) fused into the UNet.",
    "50 → 3–8 steps",
  ],
  [
    "No guidance pass",
    "The distilled model doesn't need classifier-free guidance, so each step is one UNet pass instead of two.",
    "2× per step",
  ],
  ["Half precision", "FP32 → FP16 for every model.", "4.27 → 2.13 GB"],
  [
    "INT8 weights",
    "Weight-only INT8 quantisation of the UNet for the Lite tier (optimum-quanto).",
    "2.13 → 1.11 GB",
  ],
  ["Tiny decoder", "TAESD replaces the 50M-parameter VAE decoder.", "−31% generation time"],
  [
    "Lower resolution",
    "Lite generates at 384 px and upscales with classical CV.",
    "−16% generation time",
  ],
];

function BigStat({
  label,
  value,
  note,
  accent = false,
}: {
  label: string;
  value: string;
  note?: string;
  accent?: boolean;
}) {
  return (
    <motion.div
      variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } }}
      whileHover={{ y: -3 }}
      className="darkroom-glass rounded-2xl p-4 transition-colors hover:border-darkroom-violet/40 sm:p-5"
    >
      <div className="text-sm text-muted-foreground">{label}</div>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div
          key={value}
          initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.35 }}
          className={`mt-1.5 font-display text-2xl font-semibold tabular-nums sm:text-3xl ${accent ? "darkroom-gradient-text" : ""}`}
        >
          {value}
        </motion.div>
      </AnimatePresence>
      {note && <div className="mt-1 text-sm text-muted-foreground">{note}</div>}
    </motion.div>
  );
}

function Panel({
  title,
  icon: Icon,
  hint,
  children,
  defaultOpen = true,
}: {
  title: string;
  icon: typeof Activity;
  hint?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  // collapsible on phones to shorten the scroll; always open on large screens
  const [open, setOpen] = useState(defaultOpen);
  return (
    <motion.section
      initial={{ opacity: 0, x: 18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
      className="darkroom-glass rounded-3xl p-5"
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left lg:pointer-events-none"
      >
        <h2 className="flex items-center gap-2.5 font-display text-lg font-semibold">
          <Icon className="h-5 w-5 text-darkroom-violet" />
          {title}
        </h2>
        <span className="flex items-center gap-2">
          {hint && (
            <span className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-muted-foreground">
              {hint}
            </span>
          )}
          <ChevronDown
            className={`h-5 w-5 text-muted-foreground transition-transform lg:hidden ${open ? "rotate-180" : ""}`}
          />
        </span>
      </button>
      <div className={`${open ? "mt-3 block" : "hidden"} lg:mt-3 lg:block`}>{children}</div>
    </motion.section>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 mt-4 text-sm font-medium text-muted-foreground first:mt-0">{children}</div>
  );
}
