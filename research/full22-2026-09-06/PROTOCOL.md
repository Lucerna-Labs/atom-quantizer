**Full 22-candidate experiment protocol — 2026-09-06**

Objective: find the best measured footprint/quality tradeoffs across all 22 candidates, including useful compositions. Report a Pareto frontier and quality bands rather than claim a universally optimal bit budget. Count complete archive bytes, decoded resident state, and measured runtime where applicable.

The v0.2.4 foundation artifacts and research history remain baselines. New research representations use a separate, explicitly versioned format with actual packed codes, stored decoder state, checksums, and standalone GGUF export. Every numerical score is computed from a serialized-then-decoded record. Encode-only calibration is not charged to inference state; codebooks, predictors, corrections, graph state, and restart state required for decode are charged.

Candidate screening uses actual model tensors and operator inputs. Sliced tensor experiments are labeled as screening, not full-model benchmarks. Finalists are encoded as complete models and evaluated through actual inference. Experiments also include the existing sequential architecture: original W → the exact main Composer 1 artifact W1 → experimental second-stage W2, with error measured against both W1 and W. Production acceptance floors are not silently relaxed; exploratory low-bit candidates are kept separate and must earn promotion through model-level quality evidence.

Data: stratified/random training rows for fitting; the validation split for adaptive selection; a fresh tail of the test split beyond the previously evaluated first 65,536 tokens for final evaluation. Preserve document grouping when resampling. Final task/teacher-fidelity checks include text, code, and mathematics. Keep failed and dominated outcomes.

| ID | Candidate | Required experiment |
|---|---|---|
| 1 | Observability / task geometry | Gaussian versus real-input ranking; calibration size and correlated-input controls |
| 2 | Marginal bit value | Global allocation over actual serialized candidates; measured budget/quality frontier |
| 3 | Correlation-directed feedback | Damped block-Hessian error compensation versus nearest/adjacent rounding |
| 4 | Joint rounding | Coupled coordinate rounding at fixed grid and byte count |
| 5 | Levels and clipping | Ternary, full four-level, optimized scales, and higher-bit controls |
| 6 | Coordinate symmetries | Diagonal scaling and signed orthogonal transforms with stored inverses |
| 7 | Vector/lattice geometry | Actual finite E8 lattice coding and scalar controls |
| 8 | Finite-state source coding | Restartable trellis/Viterbi encoding and blind stateful decoding |
| 9 | Complementary approximations | Additive vector codebooks and residual refinement with all streams counted |
| 10 | Low-rank error correction | Stored low-rank residual factors and rank/byte controls |
| 11 | Influential exceptions | Sensitivity/error-value extraction versus magnitude at matched exception count |
| 12 | Loss-fitted correction | No correction, norm restoration, gain, and legal weight-offset correction |
| 13 | Topology / locality | Array smoothing, real-feature graph repair, and shuffled/permuted controls |
| 14 | Conditional cell estimates | Frozen-code posterior reconstruction versus midpoint decoding |
| 15 | Bias/variance | Several stochastic/paired rounding seeds; average and worst outcome |
| 16 | Entropy / description length | Real entropy-coded records, metadata precision, and startup/seek costs |
| 17 | Adaptive resolution | Stored dyadic partitions versus fixed block sizes |
| 18 | Temporal stability | Actual Mamba recurrent operators/model, trajectories and increasing context |
| 19 | Fragile/coupled decisions | Actual attention Q/K coupling and decision margins |
| 20 | Selection uncertainty | Independent splits, paired resampling, ranking stability, sample allocation |
| 21 | Predictions and innovations | Decoded-anchor prediction of comparable tensors; random-pair controls |
| 22 | Machine bottlenecks | Actual packed decode/matmul, memory, and latency; no inference claim from file size alone |

Initial search axes: 2/3/4/6/8-bit scalar candidates; 16/32/64/128 groups; transform bypass versus signed Hadamard; small correction budgets; separate lattice, trellis, and additive families. Screen single mechanisms before selecting combinations. Add experimental variants only when failures or measurements justify them.

Report exact source/software/corpus/artifact hashes. Label every result by its actual boundary: tensor screen, real operator, full model, or runtime kernel. Incompatibility alone is not a completed experiment: use a matching real model/operator where required. No candidate is declared beneficial solely from its analogy or a synthetic control.
