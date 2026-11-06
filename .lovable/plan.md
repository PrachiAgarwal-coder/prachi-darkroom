# Darkroom

Build the uploaded brief as a usable, cinematic single-page image-generation demo.

## Experience
- Create the Darkroom workspace at `/`: animated latent canvas, image stage, prompt controls, responsive compute and pipeline rails, and matching visual system.
- Implement demo generation with prompt-aware procedural scenes, staged previews, QA/detections, processing toggles, retry behavior, history, performance comparison, and quality/cost visualization.
- Add optional WebSocket server connection, health/tier discovery, URL settings, progress/result/error handling, and clear demo/live/unreachable states.

## Technical details
- Keep the experience client-side in the existing TanStack Start index route; no database or third-party model API.
- Define palette, typography, effects, and motion in `src/styles.css`; add app-specific route metadata.
- Verify responsive rendering and the main demo-generation flow, then run relevant tests and inspect preview build diagnostics.
