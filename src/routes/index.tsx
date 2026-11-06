import { createFileRoute } from "@tanstack/react-router";
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
const DEFAULT_POST: PostSettings = { contrast: true, wb: true, sharp: true, upscale: false };
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
  const [prompt, setPrompt] = useState("");
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
    const stored = localStorage.getItem("darkroom-ws-url") ?? DEFAULT_URL;
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
              sharpness: 0.82,
              clipped_pct: 0.3,
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
      if (!prompt.trim()) {
        setPrompt("a red cube to the left of a blue sphere on a wooden table at sunset");
        return;
      }
      retryRef.current = false;
      beginGeneration(prompt.trim());
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

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-darkroom-canvas text-foreground selection:bg-darkroom-violet/30">
      <LatentField active={jobState === "loading" || jobState === "queued"} />
      <div
        className="pointer-events-none fixed inset-0 z-[1] bg-[radial-gradient(ellipse_at_50%_35%,transparent_15%,var(--darkroom-canvas)_100%)] opacity-85"
        aria-hidden
      />
      <div className="relative z-10 mx-auto max-w-[1680px] px-4 pb-8 sm:px-6 lg:px-8">
        <header className="flex min-h-[72px] items-center justify-between border-b border-darkroom-panel-edge/80 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl border border-darkroom-violet/30 bg-darkroom-violet/10 text-darkroom-violet">
              <Moon className="h-[18px] w-[18px]" />
            </div>
            <div>
              <div className="font-display text-[19px] font-semibold leading-none">
                Darkroom<span className="text-darkroom-coral">.</span>
              </div>
              <div className="mt-1 hidden text-[9px] uppercase tracking-[.2em] text-muted-foreground sm:block">
                On-device image generation
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <Button
              variant="outline"
              size="sm"
              className="h-9 gap-2 rounded-full border-darkroom-panel-edge bg-darkroom-panel px-3 text-[10px] font-medium sm:text-xs"
              onClick={() => {
                setUrlDraft(connectionUrl);
                setDialogOpen(true);
              }}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${connection === "live" ? "bg-darkroom-success" : connection === "unreachable" ? "bg-darkroom-coral" : connection === "connecting" ? "darkroom-gradient darkroom-pulse" : "bg-darkroom-amber"}`}
              />
              <span className="max-w-[156px] truncate">
                <span className="sm:hidden">
                  {connection === "live"
                    ? "Live"
                    : connection === "unreachable"
                      ? "Offline"
                      : connection === "connecting"
                        ? "Connecting"
                        : "Demo"}
                </span>
                <span className="hidden sm:inline">{modeLabel}</span>
              </span>
              <Settings2 className="h-3.5 w-3.5 text-muted-foreground" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              aria-label="Toggle compute and pipeline panels"
              onClick={() => setShowRails(!showRails)}
            >
              <Layers3 className="h-4 w-4" />
            </Button>
            <div className="hidden items-center gap-1.5 text-[10px] text-muted-foreground md:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-darkroom-teal" />
              LOCAL INFERENCE LAB
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-4 pb-10 pt-4 lg:grid-cols-[minmax(220px,260px)_minmax(0,1fr)_minmax(250px,300px)] lg:gap-5 lg:pb-8 lg:pt-6">
          <div
            className={`darkroom-sheet-rail flex gap-3 overflow-x-auto lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:flex-col lg:overflow-y-auto ${showRails ? "" : "max-lg:hidden"}`}
          >
            <section
              className="darkroom-sheet darkroom-glass darkroom-surface min-w-[min(82vw,300px)] p-4 lg:min-w-0 lg:p-5"
              aria-label="Compute budget"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-darkroom-violet" />
                  <h2 className="font-display text-sm font-semibold">Compute budget</h2>
                </div>
                <span className="text-[9px] uppercase tracking-[.14em] text-muted-foreground">
                  LOCAL
                </span>
              </div>
              <div className="mt-5 text-[10px] uppercase tracking-[.16em] text-muted-foreground">
                Device profile
              </div>
              <div className="mt-2 grid grid-cols-2 gap-1 rounded-xl border border-darkroom-panel-edge bg-darkroom-canvas/50 p-1">
                {(["auto", "budget", "mid", "flagship"] as DeviceKey[]).map((key) => (
                  <Button
                    key={key}
                    variant="ghost"
                    size="sm"
                    className={`h-8 min-w-0 truncate rounded-lg px-2 text-[10px] ${device === key ? "bg-secondary text-foreground" : "text-muted-foreground"}`}
                    onClick={() => setDevice(key)}
                  >
                    {DEVICE_LABEL[key]}
                  </Button>
                ))}
              </div>
              <div className="mt-4 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Memory ceiling</span>
                <span className="font-medium tabular-nums">{allowance.toLocaleString()} MB</span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary">
                <div
                  className={`h-full rounded-full transition-all ${tierDecision.downgraded ? "bg-darkroom-warning" : "darkroom-gradient"}`}
                  style={{
                    width: `${Math.min(100, (activeSpec.measured.peakMB / allowance) * 100)}%`,
                  }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[9px] text-muted-foreground">
                <span>Peak {activeSpec.measured.peakMB.toLocaleString()} MB</span>
                <span>of {allowance.toLocaleString()}</span>
              </div>
              <div className="mt-5 flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-[.16em] text-muted-foreground">
                  Compute tier
                </span>
                <span className="text-[9px] text-muted-foreground">
                  {requestedTier === "adaptive" ? "AUTO-SELECT" : "MANUAL"}
                </span>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-1.5">
                {(["adaptive", "quality", "balanced", "lite"] as const).map((key) => (
                  <Button
                    key={key}
                    variant="ghost"
                    size="sm"
                    aria-pressed={requestedTier === key}
                    className={`h-9 justify-start rounded-lg px-2.5 text-[11px] ${requestedTier === key ? "darkroom-active" : "border border-darkroom-panel-edge bg-darkroom-canvas/40 text-muted-foreground hover:text-foreground"}`}
                    onClick={() => setRequestedTier(key)}
                  >
                    <span
                      className={`mr-1.5 h-1.5 w-1.5 rounded-full ${requestedTier === key ? "bg-primary-foreground" : "bg-muted-foreground/40"}`}
                    />
                    {key[0]?.toUpperCase()}
                    {key.slice(1)}
                  </Button>
                ))}
              </div>
              {tierDecision.downgraded && (
                <div className="mt-3 flex gap-2 rounded-lg border border-darkroom-warning/25 bg-darkroom-warning/10 p-2.5 text-[10px] leading-relaxed text-darkroom-warning">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>
                    {TIER_LABEL[tierDecision.requested ?? activeTier]} exceeds this device’s memory
                    allowance. Using {TIER_LABEL[activeTier]} instead.
                  </span>
                </div>
              )}
              <div className="mt-4 border-t border-darkroom-panel-edge pt-3">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-muted-foreground">{TIER_LABEL[activeTier]} spec</span>
                  <span className="tabular-nums text-foreground">
                    {activeSpec.steps} steps · {activeSpec.res}px
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-y-1.5 text-[9px]">
                  <span className="text-muted-foreground">Denoiser</span>
                  <span className="truncate text-right">{activeSpec.unet}</span>
                  <span className="text-muted-foreground">Precision</span>
                  <span className="text-right">{activeSpec.precision}</span>
                  <span className="text-muted-foreground">Decoder</span>
                  <span className="text-right">{activeSpec.decoder}</span>
                </div>
              </div>
              <div className="mt-5 border-t border-darkroom-panel-edge pt-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-[.14em] text-muted-foreground">
                    Post-processing
                  </span>
                  <WandSparkles className="h-3.5 w-3.5 text-darkroom-coral" />
                </div>
                {(
                  [
                    ["contrast", "Adaptive contrast"],
                    ["wb", "White balance"],
                    ["sharp", "Edge sharpening"],
                    ["upscale", "Upscale to 512"],
                  ] as const
                ).map(([key, label]) => (
                  <div
                    key={key}
                    className="flex min-h-8 items-center justify-between gap-3 text-[10px]"
                  >
                    <span>{label}</span>
                    <Switch
                      checked={post[key]}
                      onCheckedChange={(checked) => setPost((old) => ({ ...old, [key]: checked }))}
                      aria-label={label}
                    />
                  </div>
                ))}
              </div>
            </section>

            <section
              className="darkroom-sheet darkroom-glass darkroom-surface min-w-[min(82vw,300px)] p-4 lg:min-w-0 lg:p-5"
              aria-label="Generation pipeline"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-darkroom-teal" />
                  <h2 className="font-display text-sm font-semibold">Live pipeline</h2>
                </div>
                <span
                  className={`rounded-full px-2 py-1 text-[8px] uppercase tracking-[.12em] ${jobState === "loading" || jobState === "queued" ? "bg-darkroom-violet/15 text-darkroom-violet" : "bg-secondary text-muted-foreground"}`}
                >
                  {jobState === "queued" ? "Queued" : jobState === "loading" ? "Active" : "Standby"}
                </span>
              </div>
              <div className="relative mt-5 space-y-0.5">
                <div className="absolute bottom-4 left-[7px] top-4 w-px bg-darkroom-panel-edge" />
                {STAGES.map((stage) => {
                  const state = stageStates[stage];
                  return (
                    <div
                      key={stage}
                      className="relative flex min-h-10 items-center justify-between gap-3 py-1.5"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className={`relative z-[1] h-[9px] w-[9px] rounded-full border ${state.state === "run" ? "darkroom-gradient darkroom-pulse border-transparent shadow-[0_0_12px_var(--darkroom-violet)]" : state.state === "ok" ? "border-darkroom-teal bg-darkroom-teal" : "border-darkroom-panel-edge bg-darkroom-canvas"}`}
                        />{" "}
                        <span
                          className={`text-[11px] ${state.state === "wait" ? "text-muted-foreground" : "text-foreground"}`}
                        >
                          {STAGE_LABEL[stage]}
                        </span>
                      </div>
                      <span className="min-w-10 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
                        {state.state === "run" ? (
                          <LoaderCircle className="ml-auto h-3 w-3 animate-spin text-darkroom-violet" />
                        ) : state.ms !== undefined ? (
                          `${Math.round(state.ms)} ms`
                        ) : (
                          "—"
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 flex items-center justify-between border-t border-darkroom-panel-edge pt-3 text-[10px]">
                <span className="text-muted-foreground">Total pipeline</span>
                <span className="font-mono tabular-nums">
                  {stageTotal > 0 ? `${Math.round(stageTotal)} ms` : "—"}
                </span>
              </div>
              {statusMessage && (jobState === "queued" || connection === "unreachable") && (
                <div className="mt-3 flex items-center gap-2 rounded-lg bg-secondary/70 p-2.5 text-[10px] text-muted-foreground">
                  <Wifi className="h-3.5 w-3.5 shrink-0" />
                  {statusMessage}
                </div>
              )}
            </section>
          </div>

          <section
            className="order-first min-w-0 lg:order-none"
            aria-label="Image generation stage"
          >
            <div className="mb-3 flex items-end justify-between gap-2 px-0.5">
              <div>
                <div className="text-[9px] uppercase tracking-[.22em] text-darkroom-violet">
                  LATENT SPACE / 01
                </div>
                <h1 className="mt-1 font-display text-xl font-semibold tracking-normal sm:text-2xl">
                  Image stage
                </h1>
              </div>
              <div className="flex items-center gap-2 text-[9px] uppercase tracking-[.13em] text-muted-foreground">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${jobState === "loading" ? "darkroom-gradient darkroom-pulse" : "bg-darkroom-teal"}`}
                />
                {jobState === "loading"
                  ? "FORMING"
                  : jobState === "queued"
                    ? "WAITING FOR GPU"
                    : "READY"}
              </div>
            </div>
            <div className="group relative mx-auto aspect-square w-full max-w-[720px] overflow-hidden rounded-[20px] border border-darkroom-panel-edge bg-darkroom-canvas shadow-[0_24px_100px_-42px_oklch(0_0_0/.9)]">
              {preview ? (
                <>
                  <img
                    className="darkroom-stage-image absolute inset-0 h-full w-full object-cover"
                    src={result?.raw && result.processed ? result.raw : preview}
                    alt={prompt ? `Generated scene: ${prompt}` : "Generated scene preview"}
                  />
                  {result?.raw && result.processed && (
                    <img
                      className="darkroom-stage-image absolute inset-0 h-full w-full object-cover"
                      src={result.processed}
                      alt="Post-processed image"
                      style={{ clipPath: `inset(0 0 0 ${compareValue}%)` }}
                    />
                  )}
                  {result?.raw && (
                    <div
                      className="absolute inset-y-0 z-[5] w-px bg-foreground/80 shadow-[0_0_12px_oklch(0_0_0/.7)]"
                      style={{ left: `${compareValue}%` }}
                    >
                      <span className="absolute left-1/2 top-1/2 grid h-8 w-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-foreground/40 bg-darkroom-canvas/80 text-foreground backdrop-blur">
                        <ArrowLeftRight className="h-4 w-4" />
                      </span>
                    </div>
                  )}
                  {result?.raw && (
                    <input
                      className="darkroom-range absolute inset-x-3 bottom-3 z-10 h-6 w-[calc(100%-24px)] cursor-ew-resize opacity-0 focus:opacity-100"
                      type="range"
                      min="0"
                      max="100"
                      value={compareValue}
                      onChange={(e) => setCompareValue(Number(e.target.value))}
                      aria-label="Compare raw and processed image"
                    />
                  )}
                  {result?.detections.map((detection, index) => (
                    <div
                      key={`${detection.label}-${index}`}
                      className="pointer-events-none absolute z-[6] border border-darkroom-teal"
                      style={{
                        left: `${detection.box[0] * 100}%`,
                        top: `${detection.box[1] * 100}%`,
                        width: `${(detection.box[2] - detection.box[0]) * 100}%`,
                        height: `${(detection.box[3] - detection.box[1]) * 100}%`,
                        opacity: showBoxes ? 1 : 0,
                        transition: "opacity 180ms",
                      }}
                    >
                      <span
                        className="absolute -top-6 left-0 flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-1 text-[9px] font-medium text-darkroom-canvas"
                        style={{
                          background: detection.ok
                            ? "var(--darkroom-teal)"
                            : "var(--darkroom-coral)",
                        }}
                      >
                        {detection.label}
                        <span className="font-mono">{Math.round(detection.score * 100)}%</span>
                      </span>
                    </div>
                  ))}
                  <div className="absolute left-3 top-3 z-[7] flex items-center gap-2 rounded-full border border-darkroom-panel-edge bg-darkroom-canvas/70 px-2.5 py-1.5 text-[9px] text-foreground backdrop-blur-xl">
                    <ImageIcon className="h-3 w-3 text-darkroom-teal" />
                    {result?.raw ? "POST-PROCESSED" : "PREVIEW FRAME"}
                  </div>
                  {result?.raw && (
                    <>
                      <span className="absolute bottom-3 left-3 z-[7] rounded bg-darkroom-canvas/70 px-2 py-1 text-[9px] text-foreground backdrop-blur">
                        RAW
                      </span>
                      <span className="absolute bottom-3 right-3 z-[7] rounded bg-darkroom-canvas/70 px-2 py-1 text-[9px] text-foreground backdrop-blur">
                        PROCESSED
                      </span>
                    </>
                  )}
                  <div className="absolute right-3 top-3 z-[7] flex gap-1.5">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 gap-1.5 rounded-full border-darkroom-panel-edge bg-darkroom-canvas/70 px-2.5 text-[9px] backdrop-blur"
                      onClick={() => setShowBoxes((old) => !old)}
                    >
                      <Crosshair className="h-3 w-3 text-darkroom-teal" />
                      {showBoxes ? "Boxes on" : "Boxes off"}
                    </Button>
                  </div>
                </>
              ) : (
                <div className="absolute inset-0 overflow-hidden">
                  <div
                    className={`absolute inset-0 bg-[radial-gradient(ellipse_at_50%_60%,oklch(0.39_0.17_285/.27),transparent_42%),radial-gradient(ellipse_at_72%_22%,oklch(0.54_0.15_38/.14),transparent_35%)] transition-opacity duration-700 ${jobState === "loading" ? "opacity-100" : "opacity-65"}`}
                  />
                  <div className="absolute inset-0 darkroom-noise-grain opacity-[.22]" />
                  <div className="absolute inset-0 grid place-items-center">
                    <div className="text-center">
                      <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-full border border-darkroom-panel-edge bg-darkroom-panel text-darkroom-violet">
                        <Sparkles className="h-5 w-5" />
                      </div>
                      <p className="font-display text-sm text-foreground/85 sm:text-base">
                        Describe a scene. It will form out of noise in a few steps.
                      </p>
                      <p className="mt-2 text-[10px] text-muted-foreground">
                        {jobState === "queued"
                          ? "Waiting for GPU capacity…"
                          : jobState === "loading"
                            ? "Latent field initializing…"
                            : "READY WHEN YOU ARE"}
                      </p>
                    </div>
                  </div>
                  <div className="absolute bottom-3 left-3 rounded-full border border-darkroom-panel-edge bg-darkroom-canvas/70 px-2.5 py-1.5 text-[9px] uppercase tracking-[.14em] text-muted-foreground backdrop-blur">
                    512 × 512 · RGB
                  </div>
                </div>
              )}
              {(jobState === "loading" || jobState === "queued") && (
                <div className="absolute bottom-3 right-3 z-20 flex items-center gap-2.5 rounded-full border border-darkroom-panel-edge bg-darkroom-canvas/80 py-1.5 pl-1.5 pr-3 backdrop-blur-xl">
                  <div className="relative grid h-8 w-8 place-items-center rounded-full bg-darkroom-panel p-[3px]">
                    <div
                      className="absolute inset-0 rounded-full darkroom-ring"
                      style={{
                        background: `conic-gradient(var(--darkroom-violet) ${(step / Math.max(1, totalSteps)) * 360}deg, var(--darkroom-panel-edge) 0deg)`,
                      }}
                    />
                    <div className="relative grid h-full w-full place-items-center rounded-full bg-darkroom-canvas text-[9px] font-semibold tabular-nums">
                      {step}/{totalSteps}
                    </div>
                  </div>
                  <span className="text-[10px] text-foreground">
                    {jobState === "queued" ? "Waiting for GPU" : "Generating"}
                  </span>
                </div>
              )}
            </div>
            {jobState === "error" && (
              <div className="mt-3 flex flex-col gap-3 rounded-xl border border-darkroom-coral/25 bg-darkroom-coral/10 p-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 gap-2.5">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-darkroom-coral" />
                  <div>
                    <div className="text-xs font-medium">Generation interrupted</div>
                    <div className="mt-1 break-words text-[10px] text-muted-foreground">
                      {errorMessage}
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 border-darkroom-panel-edge text-[10px]"
                    onClick={() => {
                      setRequestedTier("lite");
                      window.setTimeout(() => beginRef.current(prompt, seedRef.current), 0);
                    }}
                  >
                    <RefreshCw className="h-3 w-3" />
                    Retry on Lite
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-8 text-[10px]"
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
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-darkroom-warning/25 bg-darkroom-warning/10 px-3 py-2.5 text-[10px] text-darkroom-warning">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{fallbackMessage}</span>
              </div>
            )}
            <form className="darkroom-glass darkroom-surface mt-3 p-3 sm:p-4" onSubmit={submit}>
              <div className="flex items-start gap-3">
                <div className="mt-2 hidden h-8 w-8 shrink-0 place-items-center rounded-lg bg-secondary text-darkroom-violet sm:grid">
                  <WandSparkles className="h-4 w-4" />
                </div>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handlePromptKey}
                  placeholder="Describe the image you want to bring into focus…"
                  aria-label="Image prompt"
                  className="min-h-[72px] resize-none border-0 bg-transparent px-1 py-2 text-sm shadow-none placeholder:text-muted-foreground/70 focus-visible:ring-0"
                />
                <Button
                  type="submit"
                  disabled={jobState === "loading" || jobState === "queued"}
                  className="darkroom-active mt-1 h-10 shrink-0 gap-2 rounded-xl px-4 text-xs font-semibold"
                >
                  <span className="hidden sm:inline">
                    {jobState === "loading"
                      ? "Generating…"
                      : jobState === "queued"
                        ? "Queued…"
                        : "Generate"}
                  </span>
                  <Send className="h-4 w-4 sm:hidden" />
                  <Sparkles className="hidden h-3.5 w-3.5 sm:block" />
                </Button>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-darkroom-panel-edge pt-2.5">
                <span className="mr-1 text-[9px] uppercase tracking-[.14em] text-muted-foreground">
                  Try
                </span>
                {SUGGESTIONS.map((suggestion, index) => (
                  <Button
                    type="button"
                    key={suggestion}
                    variant="ghost"
                    size="sm"
                    className="h-7 max-w-full rounded-full border border-darkroom-panel-edge px-2.5 text-[9px] text-muted-foreground hover:border-darkroom-violet/40 hover:text-foreground"
                    onClick={() => setPrompt(suggestion)}
                  >
                    <span className="truncate">
                      {["CUBE + SPHERE", "TEMPLE AT NIGHT", "CUP + BOOK", "CAT + LAMP"][index]}
                    </span>
                  </Button>
                ))}
              </div>
            </form>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 px-1 text-[9px] text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="rounded-full border border-darkroom-panel-edge bg-darkroom-panel px-2 py-1 font-mono text-[9px]">
                  {TIER_LABEL[result?.tier ?? activeTier]} ·{" "}
                  {tiers[result?.tier ?? activeTier].steps} steps ·{" "}
                  {tiers[result?.tier ?? activeTier].precision} · {latency.toFixed(2)} s
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 gap-1 rounded-full px-2 text-[9px] text-darkroom-teal"
                  onClick={() => setShowBoxes((old) => !old)}
                >
                  <Crosshair className="h-3 w-3" />
                  {showBoxes ? "Detections on" : "Detections off"}
                </Button>
              </div>
              <span className="tabular-nums">
                {mounted ? `Seed ${seedRef.current.toString().padStart(5, "0")}` : ""}
              </span>
            </div>

            <section className="mt-7" aria-label="Generation quality metrics">
              <div className="mb-3 flex items-end justify-between">
                <div>
                  <div className="text-[9px] uppercase tracking-[.22em] text-darkroom-coral">
                    TELEMETRY / 02
                  </div>
                  <h2 className="mt-1 font-display text-lg font-semibold">Quality & cost</h2>
                </div>
                <span className="text-[9px] text-muted-foreground">LIVE SNAPSHOT</span>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {[
                  {
                    label: "CLIP similarity",
                    value: result ? result.metrics.clip : activeSpec.measured.clip,
                    suffix: "",
                    icon: Gauge,
                    detail: result ? "Prompt alignment" : "Tier average",
                    digits: 3,
                  },
                  {
                    label: "Latency",
                    value: result
                      ? result.metrics.total_ms / 1000
                      : activeSpec.measured.gen_ms_mean / 1000,
                    suffix: "s",
                    icon: Timer,
                    detail: `${baselineFast.toFixed(1)}× faster than baseline`,
                    color: "text-darkroom-coral",
                    digits: 2,
                  },
                  {
                    label: "Peak memory",
                    value: result ? result.metrics.peak_mb : activeSpec.measured.peakMB,
                    suffix: " MB",
                    icon: MemoryStick,
                    detail: `Allowance ${allowance.toLocaleString()} MB`,
                  },
                  {
                    label: "Model size",
                    value: result?.metrics.model_mb ?? activeSpec.measured.sizeMB,
                    suffix: " MB",
                    icon: Layers3,
                    detail: activeSpec.precision,
                  },
                  {
                    label: "Sharpness",
                    value:
                      result && typeof result.qa["sharpness"] === "number"
                        ? (result.qa["sharpness"] as number)
                        : null,
                    suffix: "",
                    icon: Maximize2,
                    detail: "Laplacian variance",
                    digits:
                      result &&
                      typeof result.qa["sharpness"] === "number" &&
                      (result.qa["sharpness"] as number) < 10
                        ? 2
                        : 0,
                  },
                  {
                    label: "Clipped pixels",
                    value:
                      result && typeof result.qa["clipped_pct"] === "number"
                        ? (result.qa["clipped_pct"] as number)
                        : null,
                    suffix: "%",
                    icon: CircleHelp,
                    detail: "Highlight / shadow",
                    digits: 2,
                  },
                ].map((item) => (
                  <Metric key={item.label} {...item} />
                ))}
              </div>
            </section>

            <section className="mt-7" aria-label="Prompt verification">
              <div className="flex items-end justify-between gap-2">
                <div>
                  <div className="text-[9px] uppercase tracking-[.22em] text-darkroom-teal">
                    PROMPT QA / 03
                  </div>
                  <h2 className="mt-1 font-display text-lg font-semibold">
                    Does it match the prompt?
                  </h2>
                </div>
                <span
                  className={`text-[10px] font-medium tabular-nums ${failedChecks > 0 ? "text-darkroom-coral" : "text-muted-foreground"}`}
                >
                  {checks.length
                    ? `${resultChecks.length - failedChecks} of ${resultChecks.length} passed`
                    : "AWAITING IMAGE"}
                </span>
              </div>
              <div className="darkroom-glass darkroom-surface mt-3 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-darkroom-teal" />
                    <span className="text-[11px] font-medium">Semantic verification</span>
                  </div>
                  <div className="flex items-center gap-2 text-[9px] text-muted-foreground">
                    <span>Auto-retry on failure</span>
                    <Switch
                      checked={autoRetry}
                      onCheckedChange={setAutoRetry}
                      aria-label="Retry automatically when a check fails"
                    />
                  </div>
                </div>
                {checks.length ? (
                  <div className="space-y-2">
                    {checks.map((check, index) => (
                      <div
                        key={`${check.text}-${index}`}
                        className="flex items-center gap-2 text-[10px]"
                      >
                        <span
                          className={`grid h-4 w-4 shrink-0 place-items-center rounded-full ${check.ok ? "bg-darkroom-success/15 text-darkroom-success" : "bg-darkroom-coral/15 text-darkroom-coral"}`}
                        >
                          {check.ok ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}
                        </span>
                        <span className={check.ok ? "text-foreground/85" : "text-darkroom-coral"}>
                          {check.text}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[10px] text-muted-foreground">
                    Checks appear when the image is ready.
                  </div>
                )}
              </div>
            </section>
          </section>

          <div
            className={`darkroom-sheet-rail flex gap-3 overflow-x-auto lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:flex-col lg:overflow-y-auto ${showRails ? "" : "max-lg:hidden"}`}
          >
            <section
              className="darkroom-sheet darkroom-glass darkroom-surface min-w-[min(84vw,320px)] p-4 lg:min-w-0 lg:p-5"
              aria-label="Race comparison"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Timer className="h-4 w-4 text-darkroom-coral" />
                  <h2 className="font-display text-sm font-semibold">The race</h2>
                </div>
                <span className="text-[9px] uppercase tracking-[.14em] text-muted-foreground">
                  ONE GPU
                </span>
              </div>
              <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">
                Optimized pipeline versus full Stable Diffusion 1.5.
              </p>
              <div className="mt-4 grid grid-cols-2 gap-2">
                {[
                  {
                    key: "current",
                    label: TIER_LABEL[activeTier],
                    sub: `${activeSpec.steps} steps · ${activeSpec.precision}`,
                    image: racePreview,
                    time: raceClock.current ?? raceTimes?.current,
                    mem: raceMem?.current,
                    active: true,
                  },
                  {
                    key: "base",
                    label: "Baseline",
                    sub: "SD 1.5 · FP32 · 50 steps",
                    image: baselinePreview,
                    time: raceClock.baseline ?? raceTimes?.baseline,
                    mem: raceMem?.baseline,
                    active: false,
                  },
                ].map((entry) => (
                  <div
                    key={entry.key}
                    className="overflow-hidden rounded-xl border border-darkroom-panel-edge bg-darkroom-canvas/45"
                  >
                    <div className="relative aspect-square bg-secondary/40">
                      {entry.image ? (
                        <img
                          src={entry.image}
                          alt={`${entry.label} race preview`}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="grid h-full place-items-center text-muted-foreground">
                          <ImageIcon className="h-6 w-6 opacity-30" />
                        </div>
                      )}
                      <span
                        className={`absolute left-2 top-2 rounded-full px-2 py-1 text-[8px] ${entry.active ? "darkroom-active" : "bg-darkroom-panel text-muted-foreground"}`}
                      >
                        {entry.active ? "CURRENT" : "BASELINE"}
                      </span>
                    </div>
                    <div className="p-2.5">
                      <div className="truncate text-[10px] font-medium">{entry.label}</div>
                      <div className="mt-0.5 truncate text-[8px] text-muted-foreground">
                        {entry.sub}
                      </div>
                      <div className="mt-3 font-display text-lg font-semibold tabular-nums">
                        {entry.time ? `${entry.time.toFixed(2)}s` : "—.—s"}
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[8px] text-muted-foreground">
                        <span>Peak memory</span>
                        <span className="tabular-nums">
                          {entry.mem ? `${Math.round(entry.mem).toLocaleString()} MB` : "—"}
                        </span>
                      </div>
                      <div className="mt-1 h-1 overflow-hidden rounded-full bg-secondary">
                        <div
                          className={`h-full ${entry.active ? "darkroom-gradient" : "bg-darkroom-dim"}`}
                          style={{ width: `${Math.min(100, ((entry.mem ?? 0) / 6500) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {raceState === "done" && raceTimes && (
                <div className="mt-3 rounded-lg border border-darkroom-teal/20 bg-darkroom-teal/10 p-2.5 text-center text-[10px] text-darkroom-teal">
                  <span className="font-semibold">
                    {(raceTimes.baseline / Math.max(0.01, raceTimes.current)).toFixed(1)}× faster
                  </span>{" "}
                  with{" "}
                  <span className="font-semibold">
                    {((raceMem?.baseline ?? 1) / Math.max(1, raceMem?.current ?? 1)).toFixed(1)}×
                    less memory
                  </span>
                </div>
              )}
              <Button
                className="darkroom-active mt-3 h-9 w-full gap-2 rounded-lg text-[11px]"
                onClick={runRace}
                disabled={raceState === "running"}
              >
                <Play className="h-3.5 w-3.5" />
                {raceState === "running" ? "Race in progress…" : "Start the race"}
              </Button>
              {raceState === "running" && (
                <div className="mt-2 flex items-center justify-center gap-1.5 text-[9px] text-muted-foreground">
                  <LoaderCircle className="h-3 w-3 animate-spin" />
                  Current tier first, baseline second
                </div>
              )}
              {raceError && (
                <div className="mt-2 rounded-lg border border-darkroom-coral/25 bg-darkroom-coral/10 p-2.5 text-[10px] text-darkroom-coral">
                  {raceError}
                </div>
              )}
              {connection !== "live" && (
                <div className="mt-2 text-center text-[9px] text-muted-foreground">
                  Demo timings from the tier table. Connect a server to race for real.
                </div>
              )}
            </section>

            <section
              className="darkroom-sheet darkroom-glass darkroom-surface min-w-[min(84vw,320px)] p-4 lg:min-w-0 lg:p-5"
              aria-label="Quality versus cost chart"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-darkroom-violet" />
                  <h2 className="font-display text-sm font-semibold">Quality vs. cost</h2>
                </div>
                <span className="text-[8px] uppercase tracking-[.13em] text-muted-foreground">
                  LOG LATENCY
                </span>
              </div>
              <div className="mt-3 h-[190px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 6, right: 8, bottom: 6, left: 0 }}>
                    <CartesianGrid stroke="var(--darkroom-panel-edge)" strokeDasharray="3 5" />
                    <XAxis
                      dataKey="latency"
                      type="number"
                      scale="log"
                      domain={[
                        Math.max(10, Math.min(...chartData.map((d) => d.latency)) / 1.8),
                        Math.max(...chartData.map((d) => d.latency)) * 1.8,
                      ]}
                      tick={{ fill: "var(--darkroom-dim)", fontSize: 8 }}
                      tickFormatter={(value) => `${(value / 1000).toFixed(value < 1000 ? 2 : 1)}s`}
                      ticks={[50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter(
                        (v) =>
                          v >= Math.min(...chartData.map((d) => d.latency)) / 1.8 &&
                          v <= Math.max(...chartData.map((d) => d.latency)) * 1.8,
                      )}
                      axisLine={{ stroke: "var(--darkroom-panel-edge)" }}
                      tickLine={false}
                    />
                    <YAxis
                      dataKey="clip"
                      type="number"
                      domain={[
                        Math.floor((Math.min(...chartData.map((d) => d.clip)) - 0.006) * 100) / 100,
                        Math.ceil((Math.max(...chartData.map((d) => d.clip)) + 0.006) * 100) / 100,
                      ]}
                      tick={{ fill: "var(--darkroom-dim)", fontSize: 8 }}
                      tickFormatter={(value) => value.toFixed(2)}
                      axisLine={false}
                      tickLine={false}
                      width={30}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3", stroke: "var(--darkroom-panel-edge)" }}
                      content={({ active, payload }) =>
                        active && payload?.[0]?.payload ? (
                          <div className="rounded-lg border border-darkroom-panel-edge bg-darkroom-canvas/95 px-2.5 py-2 text-[9px] shadow-xl">
                            <div className="font-medium">
                              {TIER_LABEL[payload[0].payload.tier as TierKey]}
                            </div>
                            <div className="mt-1 text-muted-foreground">
                              {payload[0].payload.latency} ms · {payload[0].payload.clip} CLIP
                            </div>
                          </div>
                        ) : null
                      }
                    />
                    <Scatter
                      data={chartData}
                      dataKey="clip"
                      shape={(props: {
                        cx?: number;
                        cy?: number;
                        payload?: { tier?: string; memory?: number; fill?: string };
                      }) => {
                        const radius = 5 + Math.sqrt(props.payload?.memory ?? 0) / 10;
                        const selected = props.payload?.tier === activeTier;
                        return (
                          <circle
                            cx={props.cx}
                            cy={props.cy}
                            r={selected ? radius + 2 : radius}
                            fill={props.payload?.fill}
                            fillOpacity={selected ? 0.95 : 0.7}
                            stroke={selected ? "var(--foreground)" : "transparent"}
                            strokeWidth={selected ? 1.5 : 0}
                          />
                        );
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 grid grid-cols-[1.2fr_.8fr_.7fr] border-b border-darkroom-panel-edge pb-2 text-[8px] uppercase tracking-[.12em] text-muted-foreground">
                <span>Tier</span>
                <span className="text-right">Latency</span>
                <span className="text-right">CLIP</span>
              </div>
              <div className="mt-1 space-y-0.5">
                {chartData.map((row) => (
                  <div
                    key={row.tier}
                    className={`grid grid-cols-[1.2fr_.8fr_.7fr] items-center rounded py-1.5 text-[9px] ${row.tier === activeTier ? "bg-darkroom-coral/10 text-foreground" : "text-muted-foreground"}`}
                  >
                    <span className="flex items-center gap-1.5">
                      <i className="h-1.5 w-1.5 rounded-full" style={{ background: row.fill }} />
                      {TIER_LABEL[row.tier as TierKey]}
                    </span>
                    <span className="text-right tabular-nums">
                      {row.latency.toLocaleString()} ms
                    </span>
                    <span className="text-right tabular-nums">{row.clip.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </section>

            <section
              className="darkroom-sheet darkroom-glass darkroom-surface min-w-[min(84vw,320px)] p-4 lg:min-w-0 lg:p-5"
              aria-label="Recent generations"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock3 className="h-4 w-4 text-darkroom-amber" />
                  <h2 className="font-display text-sm font-semibold">Recent frames</h2>
                </div>
                <span className="text-[9px] text-muted-foreground">{history.length} / 12</span>
              </div>
              {history.length ? (
                <div className="darkroom-scrollbar mt-3 flex gap-2 overflow-x-auto pb-1 lg:grid lg:grid-cols-4">
                  {history.map((entry) => (
                    <Button
                      key={entry.id}
                      variant="ghost"
                      className="group h-auto min-w-[70px] flex-col items-start gap-1 p-0 text-left"
                      onClick={() => applyHistory(entry)}
                    >
                      <div className="relative aspect-square w-[70px] overflow-hidden rounded-lg border border-darkroom-panel-edge">
                        <img
                          src={entry.image}
                          alt={entry.prompt}
                          className="h-full w-full object-cover transition-transform group-hover:scale-105"
                        />
                      </div>
                      <span className="w-[70px] truncate text-[8px] text-muted-foreground">
                        {entry.prompt}
                      </span>
                    </Button>
                  ))}
                </div>
              ) : (
                <div className="mt-3 flex h-24 flex-col items-center justify-center rounded-xl border border-dashed border-darkroom-panel-edge text-center">
                  <ImageIcon className="h-5 w-5 text-muted-foreground/50" />
                  <span className="mt-2 text-[10px] text-muted-foreground">
                    Generated images will gather here
                  </span>
                </div>
              )}
            </section>
          </div>
        </main>

        <footer className="border-t border-darkroom-panel-edge py-5 text-center text-[9px] text-muted-foreground">
          Images come from our own optimised diffusion pipeline, not a third-party API.
        </footer>
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="darkroom-glass max-w-md border-darkroom-panel-edge bg-darkroom-canvas text-foreground sm:rounded-2xl">
          <DialogHeader>
            <DialogTitle className="font-display text-lg">Image server</DialogTitle>
            <DialogDescription className="text-left text-xs text-muted-foreground">
              Connect your optimized diffusion pipeline. Leave this blank to use the clearly
              labelled local demo.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <label
              className="text-[10px] uppercase tracking-[.14em] text-muted-foreground"
              htmlFor="darkroom-url"
            >
              WebSocket URL
            </label>
            <input
              id="darkroom-url"
              value={urlDraft}
              onChange={(event) => setUrlDraft(event.target.value)}
              placeholder="ws://localhost:8000/ws"
              className="h-10 w-full rounded-lg border border-darkroom-panel-edge bg-darkroom-canvas px-3 text-sm outline-none focus:border-darkroom-violet"
            />
            <div className="rounded-lg border border-darkroom-panel-edge bg-darkroom-panel p-3 text-[10px] leading-relaxed text-muted-foreground">
              <span className="font-medium text-foreground">Connection status:</span>{" "}
              {connection === "live"
                ? `Live · ${healthDevice || DEVICE_LABEL[resolvedDevice]}`
                : connection === "unreachable"
                  ? `Unreachable${statusMessage ? ` · ${statusMessage}` : ""}`
                  : connection === "connecting"
                    ? "Checking /health and /tiers…"
                    : "Demo mode — not model output"}
              <div className="mt-1">The server URL is saved on this device.</div>
            </div>
          </div>
          <DialogFooter className="gap-2 sm:gap-2">
            <Button
              variant="outline"
              className="border-darkroom-panel-edge"
              onClick={() => closeServer("")}
            >
              Use demo mode
            </Button>
            <Button className="darkroom-active" onClick={() => closeServer(urlDraft.trim())}>
              <Link2 className="h-3.5 w-3.5" />
              Connect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
