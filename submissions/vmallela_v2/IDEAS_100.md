# 100 ideas from VLSI, physics, EE, and applied math

Exploration space for the next round of variants on top of `vmallela_v2`.
Each idea is labeled with the discipline a proponent would come from, a
one-line mechanism, a rationale, and a confidence estimate. None of these
duplicate entries on the v2 don't-re-run list (HPWL-only filter, SA
uphill on softs, batch FD, cluster translation, micro cycles,
perturb-restart, tabu on softs, Nesterov momentum, informed Gaussian MH,
soft bigstart, Langevin smoothing init, quantum-amplitude init, harmonic
init, spectral init via numpy eigh, seed sweeps, informed-MH, (1+1)-ES,
joint hard+soft micro).

Testing policy: fast tests are ibm01 at 220 s (matching v36/v51/v80
baselines of 0.853–0.857). A variant is "interesting" only if ibm01 at
220 s is ≤ 0.848 or wall time < 200 s at ≥ 0.855. Full sweeps happen
later, not here.

Cost baseline for comparison: **v2 at seed=42, 220 s, ibm01 ≈ 0.855–0.860**
(exact number depends on jitter — see caveats in the v2 README).

---

## Discipline A — VLSI / placement (25)

### A1. Rent-parameter-weighted move ordering
Mechanism: compute Rent's exponent `p` per macro (external/internal
pin ratio) and prioritize CD on high-`p` macros first. Rationale: macros
with high `p` touch more nets and move them earlier removes HPWL gradient
for the rest. Confidence: medium-high.

### A2. RSMT cost instead of HPWL on rank-2 nets
Mechanism: for nets with ≤ 6 pins replace HPWL by rectilinear Steiner
min-tree cost (`flute` call or hand-rolled for ≤ 4 pins). Rationale:
HPWL overcounts on few-pin nets; actual wirelength is ≤ HPWL and the
optimizer follows the wrong gradient. Confidence: medium.

### A3. Multi-level coarsen / refine (METIS-style)
Mechanism: coarsen netlist via heavy-edge matching to ≤ 64 nodes,
place the coarse graph analytically, then un-coarsen projecting onto
CD. Rationale: classical multilevel is how industrial placers escape
local minima. Confidence: high for global quality, medium for
≤ 1-hour budget fit.

### A4. FM bi-partitioning as seed
Mechanism: recursive Fiduccia-Mattheyses bisection, assigning each
half to a canvas half, before push-apart. Rationale: produces a
connectivity-matched initial placement that is strictly better than
the benchmark's initial for HPWL. Confidence: medium.

### A5. Hyperedge-aware swap (vs pairwise swap)
Mechanism: choose a 3-macro cyclic permutation of macros sharing the
same high-weight net; evaluate the cyclic swap atomically. Rationale:
three-macro flips can escape 2-swap-local minima. Confidence: medium.

### A6. Pin-offset-aware HPWL
Mechanism: include each pin's position within its macro (macro center
+ pin offset) when computing HPWL, not just the macro center.
Rationale: pin offsets can flip which side of a net is "right" as a
macro moves; using center-only over-smooths. Confidence: depends on
whether the benchmark's PlacementCost already includes pin offsets —
check first.

### A7. Slack-weighted HPWL for timing
Mechanism: given a stub slack table per net (from a timing model or
uniform approximation), weight HPWL by `1 / (1 + slack_i)`. Rationale:
even if the current proxy doesn't include timing, biasing placement
toward timing-critical nets tends to reduce density around hot paths.
Confidence: low (timing is orthogonal to proxy).

### A8. Min-cost flow for pin-to-slot assignment
Mechanism: on each cycle, solve a capacitated min-cost flow matching
movable pins to nearby grid slots with edge cost = HPWL delta.
Rationale: MCMF is optimal for the assignment sub-problem and can
unlock moves CD cannot. Confidence: medium, but expensive per cycle.

### A9. Simulated annealing with TimberWolf moves
Mechanism: the full classic Sechen move set — translate, pairwise
swap, rotate (for rectangular macros) — with Metropolis acceptance
and geometric cooling. Rationale: TimberWolf is the canonical
search-based baseline; moving on top of v2 gives a direct
comparison. Confidence: medium (rotate may not apply to soft
macros).

### A10. Linear programming relaxation of HPWL
Mechanism: fix macro sizes, replace HPWL by LP slack variables, solve
to optimality with CPLEX/Gurobi or scipy.linprog, then re-legalize.
Rationale: LP-HPWL is a classical convex relaxation. Confidence: low
for the 1-hour budget on 3000+ variables.

### A11. Quadratic placement with capacitive self-term
Mechanism: quadratic HPWL (`Σ w_ij (x_i − x_j)²`) as in Kahng-Kennings
solved by preconditioned conjugate gradient. Rationale: the quadratic
form is convex and gives a high-quality seed. Confidence: medium; the
benchmark's initial is already near-quadratic, so gain is bounded.

### A12. Star-model HPWL (clique → star)
Mechanism: for large nets replace the HPWL clique by a star model with
a "virtual center" macro; move the center via weighted average of
pins. Rationale: star model is equivalent up to constant and
reduces degree to linear — CD cost per net drops. Confidence: engine
speedup, not cost improvement.

### A13. Net-decoupling via graph orientation
Mechanism: orient each hyperedge as a directed tree (driver → sinks);
prioritize CD moves along the tree. Rationale: follows natural
signal-flow direction of the netlist. Confidence: low without a
timing model.

### A14. Macro orientation / mirroring
Mechanism: enumerate 8 rotations/mirrorings per hard macro, accept if
proxy improves (and fits). Rationale: orientation changes pin
positions, which can reduce HPWL without moving macro center.
Confidence: depends on whether the benchmark allows rotation —
check `.plc` spec first.

### A15. Top-down partitioning with flow shelves
Mechanism: divide the canvas into horizontal shelves (as in shelf
placement); assign macros to shelves by FM partitioning, then pack.
Rationale: shelf packing produces legal placements by construction.
Confidence: medium; shelves work well for sea-of-gates but less for
macro-dominated floorplans.

### A16. Integer linear programming for small macro subsets
Mechanism: select a random subset of 5–10 hard macros, solve ILP on
their positions while holding others fixed. Rationale: solves the
subset exactly. Confidence: low for size > 7 due to MIP explosion.

### A17. Branch-and-bound on the densest 2 %
Mechanism: identify the 2 % of macros in the most congested regions;
B&B on their swap permutations. Rationale: congestion is the 55 %
cost component and the hottest cells drive the smoothed max.
Confidence: medium.

### A18. Timing-driven pin swapping
Mechanism: for cells with logically equivalent input pins (AND/OR
commutativity), swap pin-net assignments to reduce HPWL. Rationale:
zero-cost legal transformation that directly shrinks HPWL.
Confidence: depends on whether the benchmark exposes this — unlikely
for TILOS-style proxy.

### A19. Congestion-gradient-aware LNS
Mechanism: pick LNS destroy subset as the contiguous region of top-k
congested grid cells plus the macros occupying them. Rationale:
targets the cost component with the highest marginal gain.
Confidence: medium.

### A20. Net-bounding-box shrink heuristic
Mechanism: compute each net's current bounding box; for each pin on
the box boundary, attempt to step it toward the box center. Rationale:
moves pins off the "corners" that define HPWL. Confidence: medium —
may overlap with per-net weighted-median; quantify the delta.

### A21. Force-directed with explicit repulsion kernel
Mechanism: add a pairwise `1 / r²` repulsion between hard macros to
the force-directed step, with cutoff at 2 × macro size. Rationale:
drives overlap reduction as a continuous gradient rather than
post-hoc legalization. Confidence: medium; calibration-dependent.

### A22. Conjugate-gradient solver on quadratic subproblem
Mechanism: inside each cycle, replace one soft-CD slice with a CG
solve of `L x = b` where `L` is the net Laplacian and `b` encodes
fixed pins (the current hard positions). Rationale: CG is
superlinearly convergent on this sparse SPD system. Confidence:
medium.

### A23. Tight two-stage LP → CD handoff
Mechanism: solve LP to optimality on HPWL (ignoring congestion),
round to nearest grid point, then use as starting position for CD
on full proxy. Rationale: LP gives a global HPWL lower bound; CD
recovers the density/congestion surface. Confidence: low-medium.

### A24. Constraint propagation for overlap avoidance
Mechanism: model overlap as a scheduling constraint (interval
tree); propagate forbidden regions during CD probe selection.
Rationale: eliminates wasted moves that violate overlap. Confidence:
low payoff unless currently wasting many moves on infeasible probes.

### A25. HeAP / ePlace inspiration — electrostatic density model
Mechanism: replace squared-density penalty by Poisson-solved
electrostatic potential, use FFT-based Poisson solver. Rationale:
ePlace's key idea; well-behaved gradient vs squared penalty.
Confidence: medium, but changes the objective — can only be a
surrogate inside our exact-proxy loop.

---

## Discipline B — physics (25)

### B1. Parallel tempering (replica exchange MC)
Mechanism: run N replicas at temperatures T_1 < … < T_N, swap
configurations periodically with acceptance ∝ exp((β_i − β_j)
(E_i − E_j)). Rationale: escapes local minima without committing
the main thread to high T. Confidence: medium (needs N cores we
don't have).

### B2. Nosé-Hoover thermostat MD
Mechanism: evolve macro positions by Hamiltonian dynamics with a
friction coefficient that maintains a target temperature.
Rationale: deterministic ergodicity in phase space; well-studied
for constrained systems. Confidence: low-medium; step size
calibration is non-trivial.

### B3. Wang-Landau flat-histogram sampling
Mechanism: sample placements proportional to `1 / g(E)` where
`g(E)` is a running estimate of density of states; flattens the
histogram and eases barrier crossing. Rationale: provably escapes
exponential-suppression valleys. Confidence: medium; convergence
of `g` is slow.

### B4. Multi-canonical sampling
Mechanism: similar to Wang-Landau but with a tabulated weight
function. Rationale: well-established in spin glasses. Confidence:
medium.

### B5. Wolff cluster Monte Carlo
Mechanism: identify clusters of correlated macros (positively
correlated ΔHPWL) and flip them atomically at T-dependent
probability. Rationale: critical slowdown killer in Ising-like
systems. Confidence: medium; requires a correlation model.

### B6. Swendsen-Wang
Mechanism: similar to Wolff but flip all clusters at once.
Rationale: even faster mixing. Confidence: medium.

### B7. Kinetic Monte Carlo
Mechanism: propose each move with a rate `R_i` proportional to its
expected ΔCost; select by exponential sampling on rates.
Rationale: gives physically-motivated time evolution. Confidence:
low; the "rate" requires an estimator.

### B8. Simulated tempering
Mechanism: one replica, temperature annealed randomly with
Metropolis-Hastings on T. Rationale: cheaper than parallel
tempering, comparable quality. Confidence: medium.

### B9. Nested sampling (Skilling)
Mechanism: maintain N live configurations; replace the worst with a
new sample constrained to have cost below the current worst.
Rationale: well-defined evidence estimator; explicit exploration.
Confidence: low-medium for large N.

### B10. Ising-model mapping + exact polynomial-time solvers
Mechanism: reduce a subset of the placement problem to a QUBO and
solve with D-Wave-style annealing simulator. Rationale:
exactly-solvable subproblems can lower the overall cost.
Confidence: low; mapping overhead is large.

### B11. Topological defect annihilation for overlaps
Mechanism: model overlaps as topological defects (+1 vortex); add
an attractive potential toward opposite-sign defects, annihilating
them. Rationale: physics analogy for legalization. Confidence:
low; standard legalization already handles overlaps.

### B12. Gauge symmetry exploitation
Mechanism: identify translation-invariant subgroups of macros and
quotient the configuration space. Rationale: reduces effective
dimensionality. Confidence: low for generic benchmarks.

### B13. BCS-style pairing of highly connected macros
Mechanism: identify macro pairs with high net-weight connection;
constrain them to move together as a "Cooper pair" in CD.
Rationale: reduces effective DoF. Confidence: low-medium.

### B14. Renormalization group flow coarse-to-fine
Mechanism: block-coarsen the canvas (k × k blocks), place coarse
macros, then flow toward fine grid. Rationale: RG is the physics
analogue of multilevel placement. Confidence: medium (overlaps
with A3).

### B15. Mean-field approximation for crowd placement
Mechanism: approximate each macro's effect on its neighbors by a
self-consistent mean field; solve the fixed-point iteratively.
Rationale: standard statistical-mechanics tool. Confidence: low.

### B16. Vortex pair moves for macro rotation
Mechanism: identify macro rotations as local vortex operations;
evolve angular coordinates separately. Rationale: if the benchmark
admits rotation, this parameterization is natural. Confidence:
depends on rotation being legal.

### B17. Spin-echo analog for escape from metastable states
Mechanism: invert CD direction periodically to "echo out"
accumulated bias. Rationale: speeds mixing. Confidence: low.

### B18. Supersymmetry-inspired fermionic penalty
Mechanism: add a penalty preventing two macros from occupying same
grid cell (Pauli exclusion analog). Rationale: enforces legalization
via soft penalty. Confidence: low; overlaps with explicit check.

### B19. Hall effect for routing bias
Mechanism: bias routing (congestion estimation) by a Lorentz-force-
inspired transverse component. Rationale: may better model actual
Steiner routes. Confidence: very low.

### B20. Specific-heat spike detection for phase boundaries
Mechanism: measure cost variance vs T to detect first-order
transitions; linger at transitions during annealing. Rationale:
critical-slowdown avoidance. Confidence: low-medium.

### B21. Langevin dynamics on proxy-gradient estimate
Mechanism: finite-difference proxy gradient, evolve by
`dx = -∇E dt + √(2 T) dW`. Rationale: standard MD on non-smooth.
Confidence: low (Langevin init was already tried and failed; but
here we apply it as a refinement step, not init).

### B22. Underdamped Langevin with momentum
Mechanism: augment positions with momenta, integrate BBK or SOFM.
Rationale: underdamped can cross barriers deterministic CD can't.
Confidence: low-medium.

### B23. Hamiltonian Monte Carlo on smoothed surrogate
Mechanism: smooth the proxy via kernel regression; run HMC on the
smooth surrogate; reject/accept by exact proxy. Rationale: HMC
proposes long-range moves that local CD can't. Confidence: low;
smoothing error is hard to control.

### B24. Replica-symmetry breaking for glassy benchmarks
Mechanism: recognize that the dense benchmarks (ibm17, ibm18) may
be glassy; apply RSB-inspired multi-replica averaging. Rationale:
spin-glass toolkit. Confidence: low.

### B25. Thermodynamic integration for cost-landscape geometry
Mechanism: compute free energy as a function of temperature via
integrating ∂ ln Z / ∂ β; identify "good" basins. Rationale:
diagnostic, not optimization. Confidence: out-of-scope for cost
reduction but useful for understanding.

---

## Discipline C — electrical engineering (25)

### C1. RC-delay-weighted HPWL
Mechanism: scale each net's HPWL contribution by a fanout-dependent
RC weight. Rationale: proxy already approximates timing via HPWL;
an explicit weight can re-balance. Confidence: low without a
timing report.

### C2. Register-cluster grouping
Mechanism: identify registers with common clock; enforce a spatial
cluster during placement. Rationale: reduces clock-tree skew and
power. Confidence: low — not in proxy, but may help downstream
OpenROAD WNS.

### C3. Crosstalk-coupled net penalty
Mechanism: penalize parallel-running nets in the same routing
track. Rationale: real-world concern, not in proxy. Confidence:
low.

### C4. IR-drop-aware density
Mechanism: weight density by estimated current draw per cell.
Rationale: flags hot-current regions before placement finalization.
Confidence: low — outside proxy.

### C5. Electro-migration-aware placement
Mechanism: avoid clustering high-current drivers near narrow wires.
Rationale: reliability concern; proxy-agnostic. Confidence: low.

### C6. Thermal-aware macro spacing
Mechanism: increase nominal spacing between high-power macros.
Rationale: power density relief. Confidence: very low — no power
data.

### C7. Substrate noise isolation
Mechanism: ensure mixed-signal macros have guard-rings or
fill-isolate neighbors. Rationale: analog-digital separation.
Confidence: not applicable (no mixed-signal here).

### C8. Metal-layer-aware routing resource
Mechanism: model layer-specific capacity in congestion estimate.
Rationale: improves congestion fidelity. Confidence: low — proxy
is uniform.

### C9. Clock-mesh-aware register placement
Mechanism: align registers to pre-specified clock-mesh nodes.
Rationale: SOTA clock implementation. Confidence: very low.

### C10. Power-gating domain clustering
Mechanism: cluster macros belonging to the same power-gated island.
Rationale: saves leakage. Confidence: very low.

### C11. Hierarchical block-level placement
Mechanism: treat each connected subgraph as a rigid "block", place
blocks, then unfreeze internals. Rationale: divide-and-conquer.
Confidence: medium, overlaps with multilevel (A3).

### C12. Decoupling capacitor placement near high-switching blocks
Mechanism: insert decap cells adjacent to high-activity macros.
Rationale: PI concern. Confidence: very low — out of scope.

### C13. Pin-assignment optimization on hard macro boundaries
Mechanism: re-assign which port of a macro connects to which net if
the macro is "orientation-free" in its pinout. Rationale:
reduces HPWL without moving macro. Confidence: depends on
benchmark support.

### C14. Via-count minimization bias
Mechanism: penalize placements that force routes to change layers.
Rationale: router quality. Confidence: very low at placement
stage.

### C15. Floorplan perimeter snap
Mechanism: for macros with I/O pins on the boundary, bias toward
canvas edge. Rationale: reduces I/O wirelength. Confidence: low —
no I/O pin data.

### C16. Capacitor-network analog for soft-cluster placement
Mechanism: model std-cell clusters as RC loads; solve network
equations for minimum energy. Rationale: physically-motivated
spreading. Confidence: low.

### C17. Partition placement by switching activity
Mechanism: cluster high-activity cells together to share decap.
Rationale: power savings. Confidence: very low — no activity.

### C18. Fanout-tree balanced splitting
Mechanism: for nets with fanout > 8, split into a sub-tree of
buffers; place buffers at sub-tree centroids. Rationale: timing and
HPWL both benefit. Confidence: low — no buffer-insertion in
placer.

### C19. Wide-wire reservation for power buses
Mechanism: reserve vertical/horizontal strips of the canvas as
routing-forbidden zones, simulating power rails. Rationale:
realistic floorplan constraints. Confidence: very low.

### C20. ESD protection ring placement
Mechanism: cluster ESD cells near I/O pins. Rationale: SI/ESD
concern. Confidence: not applicable.

### C21. Multi-VT library timing bin grouping
Mechanism: group LVT/HVT cells separately; place timing-critical
on LVT. Rationale: power-timing trade. Confidence: very low — no
VT info.

### C22. Sequential-element clock-sink distance bound
Mechanism: set a max distance from each register to its clock
source; enforce as placement constraint. Rationale: CTS quality.
Confidence: very low.

### C23. Signal-integrity-informed net spacing
Mechanism: minimum spacing between victim-aggressor net pairs.
Rationale: SI concern. Confidence: very low.

### C24. Package pin-grid projection
Mechanism: bias I/O macros toward canvas edge closest to package
pin position. Rationale: package-die co-design. Confidence: not
applicable without package data.

### C25. Die-edge macro alignment
Mechanism: force large macros to align with die edges for easier
abutment. Rationale: regular floorplan aesthetics. Confidence:
very low.

---

## Discipline D — applied math / optimization (25)

### D1. Block coordinate descent with random blocks
Mechanism: each iteration pick a random subset of 5–20 macros and
jointly optimize (e.g., small gradient step on the block).
Rationale: escapes cyclic-CD local minima. Confidence: medium.

### D2. Randomized Kaczmarz for HPWL
Mechanism: view HPWL as a linear system; update pin positions via
randomized Kaczmarz iterations. Rationale: cheap per-iteration.
Confidence: low.

### D3. ADMM decomposition
Mechanism: decompose proxy into HPWL + density + congestion, solve
each separately with ADMM primal/dual updates. Rationale:
principled separation. Confidence: medium.

### D4. Frank-Wolfe / conditional gradient
Mechanism: at each iteration minimize a linear approximation of the
proxy over the feasible canvas, move a fraction toward the
minimizer. Rationale: projection-free; handles integer constraints
easily. Confidence: medium.

### D5. Proximal gradient on smoothed surrogate
Mechanism: smooth the proxy via Moreau envelope; take prox-grad
steps, correct by exact evaluation periodically. Rationale: prox
methods are state-of-the-art for non-smooth. Confidence: medium.

### D6. Cross-entropy method (CEM)
Mechanism: maintain a Gaussian over probe directions; weight top-k
samples by cost; update mean/covariance. Rationale: gradient-free
global optimizer. Confidence: medium.

### D7. CMA-ES (covariance matrix adaptation)
Mechanism: classical CMA-ES on the `(n, 2)` macro-position space.
Rationale: de facto standard for continuous BBO. Confidence:
medium; dimensionality issue for 10³ macros.

### D8. CMA-ES in PCA subspace
Mechanism: run CMA-ES in a rank-32 PCA subspace of observed good
placements. Rationale: reduces dimensionality. Confidence: medium.

### D9. Bayesian optimization with Gaussian process
Mechanism: model the proxy as a GP over a low-dim parameter vector
(cycle fractions, step sizes); use BO to tune. Rationale: sample-
efficient. Confidence: medium for hyperparameters, low for full
placement.

### D10. Multi-armed bandit for cycle allocation
Mechanism: each cycle picks operator fractions via UCB over a
cost-gain-per-second reward. Rationale: automates the manual
5/35/15/30/15 split. Confidence: medium.

### D11. Thompson sampling for probe direction
Mechanism: per-macro, maintain a Beta posterior over each of 8
probe directions' success rate; sample direction accordingly.
Rationale: cheap Bayesian exploration. Confidence: medium.

### D12. Upper confidence bound for probe selection
Mechanism: per-direction UCB1 score; pick highest. Rationale: no-
regret guarantee. Confidence: medium.

### D13. Monte Carlo tree search on move sequences
Mechanism: MCTS with UCB on move selection, rollout = 3-move
lookahead. Rationale: plans ahead. Confidence: low without a
cheap rollout.

### D14. Stochastic Frank-Wolfe with variance reduction
Mechanism: SFW with control variates based on previous iterations.
Rationale: theory-optimal. Confidence: low.

### D15. Mirror descent on simplex of probe directions
Mechanism: multiplicative weight update on 8 probe directions.
Rationale: simple and well-understood. Confidence: medium.

### D16. SVRG (variance-reduced gradient)
Mechanism: periodically compute full gradient (via finite diff),
use as control variate. Rationale: reduces noise in stochastic
steps. Confidence: low — full gradient is expensive.

### D17. Stochastic coordinate descent with importance sampling
Mechanism: sample macro index by weight ∝ recent ΔCost.
Rationale: focuses effort on promising macros. Confidence: medium.

### D18. Subgradient method with Polyak step
Mechanism: subgradient + step size `f(x) − f* / ||g||²`. Rationale:
classical non-smooth. Confidence: low — needs `f*` estimate.

### D19. Bundle method
Mechanism: maintain piecewise-linear lower-approximation of proxy;
solve convex QP for next step. Rationale: SOTA for non-smooth
convex. Confidence: low — proxy is non-convex.

### D20. Trust-region with non-smooth penalty
Mechanism: TR with quadratic model; reject step if actual < η ×
predicted. Rationale: combines CD with TR globalization.
Confidence: medium.

### D21. Iteratively reweighted least squares
Mechanism: approximate HPWL by quadratic with weights iteratively
updated by `1 / (|pin_i − net_centroid| + ε)`. Rationale: classic
weighted Laplacian approach. Confidence: medium.

### D22. Sinkhorn optimal transport to uniform density
Mechanism: Sinkhorn iterations to balance density across cells.
Rationale: structured way to spread macros. Confidence: medium.

### D23. Wasserstein gradient flow on density
Mechanism: density distribution evolves by Wasserstein gradient
flow; macro positions follow. Rationale: PDE-based spreading.
Confidence: low.

### D24. Riemannian manifold optimization on position manifold
Mechanism: model configuration as point on `(canvas)^n` manifold;
retract gradient steps onto feasible set. Rationale: respects
constraint geometry. Confidence: low — gain unclear.

### D25. Cut plane / Benders decomposition
Mechanism: separate "easy" (hard-macro) from "hard" (soft-macro)
variables; iterate between master and subproblem. Rationale:
classical MIP decomposition. Confidence: low.

---

## Summary triage

| Priority | Idea | Expected cost if implemented |
|----------|------|------------------------------|
| High     | A1 Rent-weighted ordering           | 20 LOC, 220 s ibm01 test |
| High     | A11 Quadratic placement seed        | 50 LOC (PCG)             |
| High     | D10 Multi-armed bandit allocation   | 30 LOC                   |
| High     | D11/D12 Thompson / UCB probe select | 40 LOC each              |
| High     | D22 Sinkhorn density spread         | 50 LOC                   |
| Medium   | A2 RSMT for small nets              | 60 LOC (Steiner-4)       |
| Medium   | A5 Cyclic 3-swap                    | 40 LOC                   |
| Medium   | A19 Congestion-gradient LNS         | 30 LOC (subset selector) |
| Medium   | A25 ePlace-style Poisson density    | 100 LOC (FFT solver)     |
| Medium   | B3 Wang-Landau                      | 80 LOC                   |
| Medium   | D1 Random-block CD                  | 30 LOC                   |
| Medium   | D3 ADMM decomposition               | 120 LOC                  |
| Medium   | D4 Frank-Wolfe                      | 60 LOC                   |
| Medium   | D21 IRLS                            | 50 LOC                   |
| Low      | A4 FM bisection seed                | 150 LOC                  |
| Low      | A14 Macro orientation               | need `.plc` check        |
| Low      | A22 CG on quadratic subproblem      | 50 LOC                   |
| Low      | B1 Parallel tempering               | 200 LOC + N cores        |
| Low      | B22 Underdamped Langevin            | 60 LOC                   |
| Low      | D6 CEM                              | 80 LOC                   |
| Low      | D7/D8 CMA-ES                        | 150 LOC                  |
| Low      | D9 BO on hyperparams                | 100 LOC                  |

Ideas marked "very low" / "not applicable" are documented for
completeness but not queued for implementation.

---

## Test results (to be populated)

Format per entry: `VNN — idea-ref — ibm01 proxy @ 220 s — delta vs
v51 (0.8533) — notes`.

<!-- results appended here as tests complete -->

## Round-1 fast tests (ibm01 @ 220 s, single thread, seed=42)

Baseline for comparison: v51/v80 at 220 s ≈ 0.8533 (jitter ±0.002).

| Variant | Idea | ibm01 proxy | Δ vs baseline | Wall | Verdict |
|---------|------|-------------|---------------|------|---------|
| v123 | D12 UCB1-ordered probe direction with early-accept-on-first-improving | **0.8542** | +0.0009 | 212 s | Tied / within jitter — probe-direction ordering does not meaningfully change quality at this budget. |
| v124 | A1 control — reverse macro order (ascending net-count) in soft CD | 0.8575 | +0.0042 | 213 s | Worse by a clear margin: the default "most-connected first" heuristic is real and worth ~0.004. |
| v125 | A19 LNS seed biased by sum-of-incident-net-sizes (proxy for congestion contribution) | 0.8578 | +0.0045 | 215 s | Worse. Uniform random LNS seed beats this weighting — biasing oversamples the same high-connectivity macros LNS has already repaired. |

### Takeaways from round 1

1. **The default probe-direction order is already near-optimal at our budgets.** UCB1-with-early-accept finishes in the same cost bucket (Δ=0.0009 is inside the 0.002 jitter floor), so more sophisticated bandit schemes here are unlikely to pay.
2. **Macro-order heuristic is load-bearing.** Ascending net-count lost 0.004 vs descending. This is a measurable effect that an A/B confirms — the existing implementation is not arbitrary.
3. **LNS seed-selection heuristics can hurt.** Weighting seeds by net-size incidence over-concentrates destroy sets on the same few hubs; the uniform-random baseline gives better coverage of the soft-macro configuration space.

### Round-2 candidates (implementable, not yet tested)

- **A5 three-macro cyclic swap** — 3-cycle permutation of macros sharing a high-weight net, evaluated atomically against the 2-swap. Tests whether k = 3 escapes a basin k = 2 cannot.
- **A11 + A22 quadratic placement via PCG** as an additional seed for the legalization tournament. Known to give strong HPWL-optimal seeds in classical analytical placers.
- **A25 ePlace-style Poisson density surrogate** as a probe pre-filter: reject probes whose Poisson potential increases above a threshold before calling `move_macro`. Would not change accepted moves but might prune evaluation.
- **D10 multi-armed bandit for cycle allocation** — replace the fixed 5/35/15/30/15 split by UCB over per-operator cost-gain-per-second. Tests whether the manual split is suboptimal.
- **D11 Thompson sampling probe direction** (sibling of D12): sample from Beta-Dirichlet, no explicit exploration bonus. Expected similar to D12.
- **D17 importance-weighted macro sampling** inside CD: probability of visiting macro `i` on pass `k+1` ∝ observed |ΔCost| on pass `k`. Focuses effort on recently-moving macros.
- **D21 IRLS** for the HPWL subproblem: iteratively-reweighted least-squares gives weighted-median convergence but exposes a smooth Hessian approximation for a larger step.

No round-2 tests have been run yet; they would need another 3–5 ibm01 runs at 220 s plus minor code changes each.

## Initial-placement variants (unexplored territory)

The current pipeline has only one degree of freedom on the starting
configuration: three push-apart damping settings (0.4 / 0.6 / 0.8)
applied to the benchmark's provided `.plc` initial. The legalization
tournament then picks between those three push outputs and the raw
initial. All four seeds are perturbations of the same starting point.

Classical placement has many other init strategies we have not tried.
Each can simply be added as an extra seed to the tournament — if it
wins, we gain; if it loses, the tournament keeps the current best and
we lose nothing except the time to compute it.

### IP1. Random init
Mechanism: sample each hard-macro position uniformly on the canvas,
feed to push-apart. Rationale: null-test — does the benchmark's
initial actually help, or can the placer recover from cold? Confidence:
low for quality, but the answer is a useful diagnostic.

### IP2. Quadratic (CG) init
Mechanism: build net Laplacian `L` (sparse, symmetric, PSD), solve
`L x = b_x` and `L y = b_y` with conjugate gradient, where `b` encodes
fixed-port positions (if any). Classical Kahng-Kennings /
GORDIAN-style quadratic placement. Rationale: gives the HPWL-minimum
of the quadratic relaxation — good starting point for downstream
refinement, even if it packs macros too tightly for congestion.
Confidence: medium.

### IP3. Jacobi / centroid iteration
Mechanism: `x_i ← Σ_j w_ij x_j / Σ_j w_ij` iterated to convergence.
Rationale: fixed-point of the quadratic Laplacian solve, without
needing CG. Converges in a few hundred iterations at O(pins / iter).
Confidence: medium.

### IP4. Min-cut recursive bisection (Breuer 1977)
Mechanism: FM-bisect the netlist, assign halves to canvas halves,
recurse. Rationale: classical connectivity-matched init. Confidence:
medium but requires a FM implementation (~150 LOC).

### IP5. Hilbert-curve ordered packing
Mechanism: order macros along a Hilbert space-filling curve by
attribute (size, connectivity), place in order along an interior
Hilbert path. Rationale: deterministic, preserves local neighborhood
under the curve's locality property. Confidence: low-medium.

### IP6. Grid-packed init by size
Mechanism: sort hard macros by size descending; pack on a regular
grid largest-first. Rationale: gives overlap-free start for dense
benchmarks — removes one push-apart step. Confidence: low.

### IP7. Block-center init (stress test for push-apart)
Mechanism: place every hard macro at the canvas center; push-apart
disperses. Rationale: tests whether push-apart alone can produce a
quality init from the worst-case starting point. Confidence: low but
informative.

### IP8. Ratio-cut spectral (non-numpy-eigh)
Mechanism: compute the Fiedler vector via Lanczos iteration on
`L_rw = I − D⁻¹ A`, project onto the canvas. Distinct from the
previously-tried numpy eigh which failed on structured matrices.
Rationale: Lanczos is more robust for sparse Laplacians. Confidence:
low-medium.

### IP9. Simulated-annealing warmup
Mechanism: 30 s of high-T SA on the benchmark initial, then use the
SA output as the tournament seed. Rationale: SA's uphill moves escape
the initial's local basin; the cooled state is a "globally explored"
warm start. Confidence: medium.

### IP10. Multiple pushed-init seeds (simple extension)
Mechanism: push-apart with more damping configurations — e.g., 8
instead of 3. Rationale: cheapest possible expansion of the
tournament. Confidence: low but trivial to test.

### IP11. Force-directed pre-placement
Mechanism: run force-directed (attract on nets, repel on macros) for
a fixed iteration count without legalization; use the result as a
seed. Rationale: FD finds reasonable layouts cheaply. Confidence:
medium.

### IP12. Analytical / Poisson-density (ePlace seed)
Mechanism: one-step of the ePlace electrostatic update from the
benchmark initial. Rationale: smooth density penalty gives a
well-spread starting point. Confidence: medium but expensive (FFT).

## Round-2 fast tests — initial placement (ibm01 @ 220 s)

| Variant | Idea | ibm01 proxy | Δ vs baseline (0.8533) | Wall | Verdict |
|---------|------|-------------|------------------------|------|---------|
| v126 | IP1 random init (uniform on canvas) | 1.0966 | **+0.2433** | 212 s | Much worse — push-apart cannot recover from cold random in 220 s. |
| v127 | IP7 center-collapsed init (all macros at canvas center + jitter) | 1.0778 | **+0.2245** | 214 s | Much worse — symmetric radial expansion helps slightly vs random, but still far from baseline. |
| v128 | IP6 regular-grid init, largest macros first | 1.0906 | **+0.2373** | 213 s | Much worse — structured but not netlist-aware. |
| v129 | IP10 8 push-apart damping configs (tournament expansion) | 0.8549 | +0.0016 | 326 s* | Tied (within jitter), wall unfair — the subclass double-ran the pipeline. |

*v129 wall is 326 s due to subclass re-running the parent's place() after its own seed-selection; inside that, the effective placer budget was still 220 s, so the proxy comparison is still meaningful. Wall is not.

### Takeaways from round 2

1. **The benchmark's `.plc` initial is genuinely load-bearing (∼ 0.24 of the
   cost at our fast-test budget).** Replacing it with any netlist-blind
   starting point (random, center, grid) loses ~0.22–0.24 in 220 s.
   The benchmark initial encodes prior work (e.g. RePlAce's output)
   that the search pipeline cannot reproduce in 220 s.
2. **Widening the push-apart-config tournament from 3 to 8 doesn't help.**
   At ibm01 budgets, the existing 3 damping settings already span the
   relevant basin entries; more configs just cost more legalize time.
3. **The remaining init-placement idea worth trying is a netlist-aware
   seed** — quadratic-CG (IP2), Jacobi centroid (IP3), or FM
   bisection (IP4). All would require reading the netlist connectivity
   (via the IncrementalEvaluator's `macro_nets` / `net_macros` dicts)
   before building an init. Expected payoff: if we can produce a seed
   of cost ≤ 1.0 (from netlist connectivity alone), it could match the
   benchmark's initial and potentially improve on it in regions where
   the benchmark init is suboptimal.

### Pattern across rounds 1 and 2

Seven ideas tested; zero clear wins over the v2 baseline at ibm01 @ 220 s.
Two categories of explanation:
1. **The v2 pipeline is already near-optimal at this budget.** The
   remaining improvement headroom may be too small to detect at
   220 s — it needs the full 3 300 s budget that produced the
   verified 1.1172 average.
2. **The benchmark initial + current pipeline forms a tightly coupled
   system.** Variations that perturb one component independently
   don't dominate because the pipeline has internal compensation
   (tournament picks best, adaptive scheduler handles plateau).
   To move the needle, we likely need a jointly-better initial + a
   pipeline stage that exploits it.

Recommended next steps if this line is pursued:
- Implement IP2 (quadratic-CG init) as an **additional** tournament
  seed, not a replacement for the benchmark init.
- If IP2 seed wins the tournament on > 50 % of benchmarks, it's real
  and should replace/augment the push-apart variants.
- Otherwise, invest the effort in algorithmic improvements to the soft
  refinement loop (e.g., D10 multi-armed bandit for cycle allocation,
  D21 IRLS for per-net HPWL) — these modify the high-leverage phase
  where most cost reduction happens.
