# AdaTopoNav three-factor ablation protocol

The draft contains three proposed modules, not two:

- **M — adaptive mapping:** spatial re-entry lock, state-dependent sampling,
  nearest-parent branch connection.
- **V — V-LOS:** DINOv2-based lookahead fallback.
- **C — waypoint modulation:** heading-priority nonlinear waypoint transform.

NoMaD checkpoint, context size, number of diffusion samples, selected waypoint,
robot limits, start/goal pair, random seed, simulator world, timeout, success
radius, and collision definition must remain identical within each paired
trial.

## Full factorial matrix

| ID | Map input | V-LOS | Modulation | Purpose |
|---|---|---:|---:|---|
| B | fixed-interval | off | off | unmodified deployment baseline |
| M | adaptive | off | off | adaptive mapping only |
| V | fixed-interval | on | off | V-LOS only |
| C | fixed-interval | off | on | modulation only |
| MV | adaptive | on | off | mapping and V-LOS |
| MC | adaptive | off | on | mapping and modulation |
| VC | fixed-interval | on | on | V-LOS and modulation |
| MVC | adaptive | on | on | complete system |

This 2x2x2 design supports main-effect and interaction analysis. If experiment
cost is prohibitive, the minimum defensible subset is `B`, `M`, `V`, `C`,
`MV`, `MC`, `VC`, and `MVC` should still be preferred because the modules can
interact strongly at corners.

## Code switches

- V-LOS off: run `global_planner.py` with `--disable-vlos`.
- V-LOS on: omit `--disable-vlos`.
- Modulation off: run `navigate_dynamic.py` with
  `--disable-waypoint-modulation`.
- Modulation on: omit that flag.
- Mapping off/on is selected by loading a fixed-interval/adaptive map made
  from the same synchronized exploration trajectory. Do not compare maps made
  from different teleoperation runs.

The fixed-map and adaptive-map files must have separate names. Record their
node count, edge count, image bytes, feature bytes, pose bytes, and total
storage. The current adaptive mapper creates a tree, so do not report it as a
mesh or claim an average degree above 2.

## Trials

Use paired trials. For every environment and pair index, all eight
configurations receive the same:

- start pose and orientation;
- goal pose/node;
- simulator reset state;
- seed;
- maximum duration;
- collision sensor and debounce rule.

Twenty trials per cell means:

`8 configurations x 3 environments x 20 trials = 480 runs`.

If only the L and Y environments are actually available:

`8 x 2 x 20 = 320 runs`.

Do not call starts/goals “randomized” unless seeds and sampled poses are saved.

## Metrics

Report per environment and overall:

- success count and SR, e.g. `17/20 (85%)`;
- distinct collisions per trial;
- completion time over successful trials, with successful-run count;
- travelled distance;
- SPL only when a consistent shortest-path distance is available;
- final goal distance and termination reason.

For mapping, report nodes, edges, storage, and average degree separately.
For the modulation module, additionally report path curvature or integrated
absolute angular velocity, time spent rotating in place, and completion time.
“Trajectory smoothness” must not be claimed from SR alone.

## Statistical reporting

Because trials are paired and success is binary, use McNemar tests for planned
pairwise comparisons and report confidence intervals for SR differences.
For continuous paired metrics, use a paired permutation test or Wilcoxon
signed-rank test when normality is not justified. Predeclare the primary
comparisons (`MVC` versus `B`, and each full-minus-one configuration versus
`MVC`) and correct multiple comparisons.

Do not populate the paper with the draft's 28%, 42%, 63%, or 95% values unless
they can be regenerated from committed raw logs.

## Paper table

```latex
\begin{table*}[t]
\caption{Three-factor system ablation. Each cell contains 20 paired trials.}
\centering
\begin{tabular}{lccccllll}
\toprule
Config. & M & V & C & Success & SR (\%) & SPL & Coll./trial & Time (s)\\
\midrule
B   & $\times$ & $\times$ & $\times$ & --/20 & -- & -- & -- & --\\
M   & $\checkmark$ & $\times$ & $\times$ & --/20 & -- & -- & -- & --\\
V   & $\times$ & $\checkmark$ & $\times$ & --/20 & -- & -- & -- & --\\
C   & $\times$ & $\times$ & $\checkmark$ & --/20 & -- & -- & -- & --\\
MV  & $\checkmark$ & $\checkmark$ & $\times$ & --/20 & -- & -- & -- & --\\
MC  & $\checkmark$ & $\times$ & $\checkmark$ & --/20 & -- & -- & -- & --\\
VC  & $\times$ & $\checkmark$ & $\checkmark$ & --/20 & -- & -- & -- & --\\
MVC & $\checkmark$ & $\checkmark$ & $\checkmark$ & --/20 & -- & -- & -- & --\\
\bottomrule
\end{tabular}
\end{table*}
```
