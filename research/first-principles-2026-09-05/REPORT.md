**Atom Quantizer: first principles that may improve the project**

**Historical snapshot:** this study describes revision `b96d235`. Subsequent v0.2.4 work implements the four initial foundations; see [the implementation and benchmark](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/docs/FOUNDATIONS.md). The diagnostic source and evidence below are retained from the original revision. Reproduce that historical diagnostic against that revision, not the modified current modules.

Study date: 2026-09-05. Source revision: `b96d235629e39a3eda27d340557946c080ee01a2`, version 0.2.3. Scope: this repository, its companion Q2 crate, and relevant primary literature. The codec implementation was not changed. This note and the accompanying diagnostic are study artifacts.

**My main finding:** the most promising extension is to make atoms account for the *functional importance of an error*, the *complete information needed to decode*, and the *actual cost of their composition*. This supports the existing PRE → EXTRACT → QUANTIZE → POST → VERIFY architecture. It also makes more ambitious Q2 mechanisms testable without repeating the earlier KL-only failure.

There are 22 candidate principles below. “New” means absent from the inspected implementation, not a claim of scientific novelty. “Extension” means a materially different use of an existing atom or roadmap item. Published results support trying a mechanism; they do not establish that it improves Atom Quantizer. The more exploratory connections include graph topology, Bayesian dequantization, temporal stability, and predictive coding across related tensors.

**What the project currently does**

The normal CLI selects one Q2/Q4/Q8 strategy per tensor, using block-specific scales. Eligible matrices receive rowH8, symmetric or affine block quantization, inverse rowH8, and row-norm restoration. Candidate measurements combine sampled output KL with tensor cosine. The symmetric codec can tune a five-tap repair on its encoded representation. The discovery kit already includes DCT, Haar rotations, wavelet repair, μ-law, residual carry, Lloyd–Max codebooks, outlier extraction, bulk/tail routing, superposition, and nested scales. Those are not being relabeled as new discoveries here.

The companion crate keeps the explicit sequential architecture W → Composer 1 → W1 → Composer 2 → W2. Its current Composer 1 helper is an approximation of the main picker: it checks a 0.99 cosine threshold before norm restoration, omits KL, and does not receive the protected-tensor name policy. Its results therefore need their own baseline identity. See [main picker](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/main.rs:82) and [companion helper](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/crates/atom-quantizer-q2/src/q2_composer.rs:565).

The README correctly labels the project under development. Its Qwen3.5-0.8B result is a recorded v0.2.2 snapshot, including a manual inference check, rather than a fresh v0.2.3 evaluation. I found no model weights or calibration datasets in this repository outside the new synthetic fixture. I did not run a real-model benchmark, CUDA evaluation, perplexity test, or inference engine during this study.

**Verified foundations that affect every proposed improvement**

All 27 existing CPU tests passed with `cargo test --workspace --all-targets`. Neither executable currently has unit tests of its orchestration. A separate, small diagnostic imported the unchanged production modules and exercised the real main CLI on a 4×32 F32 tensor. Its results are in [evidence.json](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/research/first-principles-2026-09-05/evidence.json).

| Observation | Evidence and consequence |
|---|---|
| The scored reconstruction differs from the stored codec's decode. | The CLI accepted Q4 at cosine **0.9902931**. Production `wq::expand` on its serialized OQ02 record gave **0.06848829** against the original. Applying inverse rowH8 and original row norms made the decode identical to the F32 passthrough output, with maximum absolute difference **0**. This is a synthetic boundary reproduction, not a model accuracy estimate. |
| Decode side information is incomplete. | `CompressedTensor` and OQ02 carry no transform identifier, row shape, or norm restoration data. The main CLI writes the encoded `cand` to OQ02 but measures and writes `post_recon` to the passthrough GGUF. See [encoded state](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/wq.rs:123), [OQ writer](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/oq.rs:57), and [writeback](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/main.rs:441). |
| Integrity covers codes, not the complete record. | Multiplying a serialized scale by 1.25 changed the decoded weights while production `oq::verify_file` still returned `all_verified=true`. `bytes_tag` hashes packed bytes only and maintains a 32-bit state, even though it returns 64 bits. Transform metadata and scales need coverage if verification is to certify the decoded record. See [tag](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/wq.rs:152). |
| Norm restoration does not repair a row's angle. | Positive rescaling leaves cosine unchanged. The probe measured **0.91381156 before and after**. It can change whole-tensor cosine through relative row weighting, but that is not guaranteed to help. The discovery pipeline also restores norms before reinserting outliers and undoing row DC; one example ended with norm **125.13581** against a target of **99.15644**. See [stage order](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/stacks.rs:899). |
| Some search costs describe hypothetical encodings. | A `SuperposeSymAsym` block produced **22 distinct reconstructed values**, while the search budgets one nominal four-bit stream plus metadata: **7 bits/weight**. A straightforward encoding of the implemented two reconstructions needs two code streams, giving **11 bits/weight** with those f32 scales/minimum. A different joint encoding might reduce this, but it must be specified and measured. Row norms are uncharged; some side data are used at f32 while charged as f16. See [search accounting](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/stacks.rs:885). |
| Failure to meet the gates is not always rejection. | After all candidates fail, the main picker returns Q8 anyway, and the caller can serialize it with a warning. A feasible raw/F16/BF16 escape candidate or explicit `NoAcceptableCandidate` outcome would distinguish verified acceptance from best-effort output. See [picker fallback](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/main.rs:144). |
| The Q2 export is not the selected search result. | The companion's writeback branch hardcodes `rowDC + top4 + refractQ2 + preserve`; it does not export the measured winner or require that this stack pass all gates. Its bit estimator also omits parts of superpose/refract/norm state. See [export](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/crates/atom-quantizer-q2/src/main.rs:238) and [cost estimator](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/crates/atom-quantizer-q2/src/q2_composer.rs:342). |

The underlying first principle is **decoder sufficiency**: every operation that determines the accepted reconstruction must be reproducible from the stored record and declared shared state, without access to the original weights. The acceptance check should consume the same serialized-then-decoded object that a consumer receives. Model metadata, transform state, sidecars, rounding of side data, and headers all belong in the cost. The current passthrough GGUF exercises reconstructed values at the source dtype; its unchanged file size does not demonstrate compressed inference.

**1. Observability and task-weighted geometry — system identification and information geometry**

**Principle.** An instrument cannot detect changes in its null space. Let E = Ŵ − W and let X contain the calibration inputs. Any nonzero E with EX = 0 is invisible to those inputs. Four probes in d > 4 input dimensions leave unprobed directions even before softmax. Independently, softmax(y + c·1) = softmax(y), so its KL cannot observe common-mode output shifts. The diagnostic gave KL = 0 in both directions for a shift with raw output MSE = 16.

For input second moment Cx = E[xxᵀ], the exact expected linear output error is:

`Dlinear = E[||Ex||²] = tr(E Cx Eᵀ)`.

Cx includes the mean contribution; it is not only centered covariance. The current sampler approximates Cx = I/d because it scales standard Gaussian coordinates by 1/√d. For a row w, its sampled output variance is ||w||²/d. Small logits can make softmax KL numerically tiny without establishing functional fidelity.

**Atom.** Extend VERIFY with real activation samples, unsoftmaxed output error, and operator-specific measurements. Add a regularized block second-moment sketch when full samples are too expensive. For downstream sensitivity, optionally use a positive semidefinite output weighting G and evaluate `tr(G E Cx Eᵀ)` as an approximation when a fixed G is appropriate. Actual token-logit KL has different semantics from softmax over arbitrary hidden features.

**Status and cost.** Real calibration is already on the roadmap; observability checks, covariance geometry, and downstream weighting are extensions. Encode-time cost ranges from channel statistics to block matrices or captured activations. A diagonal importance matrix cannot reconstruct activation samples or off-diagonal correlations. GPTQ provides a direct precedent for fitting layer outputs using calibration inputs. [GPTQ](https://arxiv.org/abs/2210.17323)

**First test / reject if.** Compare Gaussian KL, real-input linear error, and block-output error against held-out token loss for the same candidate reconstructions. Reject a more expensive proxy if it does not improve candidate ranking or detect failures missed by the present gates. Include common-mode shifts, correlated features, scale errors, and held-out probe directions as negative controls.

**2. Marginal value of a bit — rate–distortion theory and operations research**

**Principle.** Under a shared resource budget, spend the next bit where it removes the most harmful error. For candidate j at group g, measure distortion Dgj and complete storage Bgj, then select:

`min Σg Dg,j(g), subject to Σg Bg,j(g) ≤ Btotal`.

This separable model is a useful approximation; cross-layer interactions still require whole-model checking. A Lagrangian selector minimizes `Dgj + λBgj`. With compute included, use `D + λB + μT`, where T is measured cost on the target workload. Gaussian reverse water-filling is a theoretical special case, not a guarantee for low-bit LLM weights. [Rate–distortion treatment](https://web.stanford.edu/class/ee398a/BookWiegandSchwarz.pdf)

**Atom.** A budget allocator around the existing candidate composer. Keep a Pareto frontier per tensor initially; later permit row or block groups. Candidate choices can include no transform, alternative transforms, Q2/Q4/Q8, and a high-precision escape. HAWQ-V2 supplies a neural-network precedent for sensitivity-guided mixed precision and Pareto selection. [HAWQ-V2](https://arxiv.org/abs/1911.03852)

**Status and cost.** New global orchestration; moderate calibration/search cost, strategy-map bytes, and possible kernel fragmentation. It addresses the current tensor-wide decision and the search winner chosen by cosine alone.

**First test / reject if.** Fix total *serialized* bytes and compare whole-model loss against the current picker. Reject improvements that consume extra hidden side data or disappear at matched size and acceptable latency.

**3. Route residuals through correlated coordinates — feedback control and numerical optimization**

**Principle.** Compensation should follow functional coupling. A quantization error in coordinate i can be offset by changes to other coordinates correlated with it under the input distribution. Sending every residual to the next flattened weight assumes a coupling that may not exist.

**Atom.** Add an encode-time QUANTIZE candidate using a damped block Hessian or second-moment factorization. After committing a code, redistribute its error among the remaining coordinates according to that geometry. The final artifact contains ordinary codes and scales; the calibration matrix can remain encode-only. This is materially different from both adjacent residual carry and re-quantizing an already decoded tensor in a POST stage. GPTQ gives a concrete second-order compensation precedent. [GPTQ algorithm](https://arxiv.org/html/2210.17323v2)

**Status and cost.** Extension of error feedback; strong external precedent. Factorization and updates cost encode time and temporary memory. Regularization matters when calibration matrices are nearly singular.

**First test / reject if.** Compare nearest rounding, current one-dimensional feedback, and correlation-directed compensation with identical code layouts and held-out inputs. Reject if improvements are confined to the calibration batch or if damping sensitivity makes decisions unstable.

**4. Joint rounding as an energy minimization — discrete optimization and statistical mechanics**

**Principle.** Individually nearest values need not minimize a coupled system's error. With rounding choices zᵢ ∈ {0,1}, local distortion can contain interactions `zᵀAz + bᵀz`. These are the same mathematical interactions used in binary energy and Ising-style models.

**Atom.** A small joint-rounding solver inside QUANTIZE: coordinate descent or a bounded beam search over round-up/round-down choices, scored by activation error. AdaRound is a concrete precedent for task-adapted rounding using a relaxed binary optimization. [AdaRound](https://arxiv.org/abs/2004.10568)

**Status and cost.** New candidate; no additional code bits if it uses the same grid, but additional encode-time search. This borrows an optimization structure from statistical mechanics; it does not require a physical annealer.

**First test / reject if.** Test groups of 8–32 weights at fixed scales, compare with nearest rounding and second-order feedback, then allow scale tuning separately. Reject the solver if it cannot beat simpler compensation after held-out validation at equal storage.

**5. Allocate reconstruction levels and clipping jointly — scalar coding and decision theory**

**Principle.** Covering the maximum magnitude exactly need not minimize total distortion. Scale selection balances overload error in the tails against resolution in the bulk. For fixed integer assignments q and weights hᵢ, a least-squares scale is `s* = Σhᵢwᵢqᵢ / Σhᵢqᵢ²`; assignments and scale can be alternated, with clipping candidates evaluated explicitly.

**Atom.** Extend Q2 with optimized ternary scale/clipping and an optional genuine four-level symmetric alphabet. Current symmetric Q2 packs two bits but emits three reconstruction levels, {-s,0,s}; `--asym` uses four levels with an extra minimum. A shared four-level shape such as {-a,-b,b,a}, normalized to one block scale, makes a different zero-versus-resolution tradeoff without automatically requiring an affine offset. See [current levels](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/src/wq.rs:68).

**Status and cost.** Extension of the existing scalar-codebook work into the live Q2 bottleneck. Small encode-time searches, optional codebook identifier or ratio metadata. This is a candidate family, not a claim that four levels always beat a ternary alphabet.

**First test / reject if.** Compare max-abs ternary, optimized ternary, affine four-level, and fixed/shared nonuniform four-level candidates at complete byte parity. Include zero-heavy tensors, heavy tails, and protected vectors. Reject if zero loss, overload, or metadata outweighs the finer alphabet.

**6. Exploit exact coordinate symmetries — linear algebra, preconditioning, and gauge freedom**

**Principle.** A function can have many equivalent coordinate representations, with very different quantization difficulty. For invertible S, `Wx = (WS)(S⁻¹x)`. For an orthogonal R, `Wx = (WR)(Rᵀx)`.

**Atom.** Add calibration-guided diagonal scaling and signed/permuted orthogonal transforms. The codec-only form stores enough state to undo the representation at decode. A fused runtime form changes the corresponding inputs or neighboring operators too. Diagonal scaling and randomized Hadamard processing have direct precedents. [SmoothQuant](https://arxiv.org/abs/2211.10438), [QuIP#](https://arxiv.org/abs/2402.04396)

**Status and cost.** Extension beyond fixed rowH8. Store scales/signs/permutation or a precisely specified deterministic transform; account for application cost. A generic scaling cannot pass through an arbitrary nonlinearity unchanged. Residual additions and tied weights also constrain legal transformations.

**First test / reject if.** First verify the unquantized transformed function against the original. Then compare quantized held-out outputs at equal total bytes. Reject transformations that merely move error into the inverse, require unavailable decoder state, or add more runtime work than their compression saves.

**7. Quantize vectors, not independent coordinates — geometry and lattice packing**

**Principle.** A fixed number of representable vectors can cover a multidimensional distribution more efficiently than a Cartesian product of scalar levels. The codebook shape and the source distribution both matter.

**Atom.** A small 4D or 8D vector quantizer after an eligible rotation, with a shared structured lattice codebook. This is different from the existing per-block one-dimensional Lloyd–Max codebook. QuIP# demonstrates an LLM implementation combining randomized Hadamard processing, block rounding, and an E8-based codebook. Its result motivates an atom experiment, not a promise of equivalent Q2 quality here. [QuIP#](https://arxiv.org/abs/2402.04396)

**Status and cost.** New QUANTIZE family with strong LLM precedent. Codebook identifiers, group scale, lookup layout, and a compatible decoder are required. Local eight-wide Hadamard alone does not guarantee the same distribution or assumptions as the published method.

**First test / reject if.** Compare scalar and vector codebooks with matched total bytes, calibrated output error, and actual decode throughput. Reject if codebook traffic or nearest-vector search destroys the useful advantage.

**8. Use finite decoder state to enlarge the code space — communications and dynamic programming**

**Principle.** A bounded-state decoder can represent long vector patterns without storing an exponentially large independent lookup table. A trellis encodes choices as paths through states.

**Atom.** A trellis-coded QUANTIZE candidate with independent restartable chunks. Use bounded dynamic programming at encode time and a specified state transition at decode. QTIP transfers this communication-coding mechanism to LLM weight quantization. [QTIP](https://arxiv.org/abs/2406.11235)

**Status and cost.** New, more ambitious candidate. Budget transition information, chunk start state, restart points, and any tables. Source coding here improves approximation; it is not corruption correction.

**First test / reject if.** Compare against scalar and vector coding on the same activation traces, including random row access and a damaged-chunk test. Reject if sequential dependencies, launch overhead, or recovery cost dominate the gain.

**9. Encode complementary approximations — information retrieval and successive refinement**

**Principle.** Several small codebooks can generate many useful reconstructions through their sums. A second code should explain residual structure left by the first; averaging two independently complete quantizations is a different optimization.

**Atom.** An additive-codebook QUANTIZE candidate, `ŵ = c₁[k₁] + c₂[k₂]`, or a base-plus-enhancement record. AQLM is a direct precedent for adapting additive quantization from information retrieval to layer outputs. [AQLM](https://arxiv.org/abs/2401.06118)

**Status and cost.** New representation related to the existing superpose atom. Both indices, all codebooks, scales, and any layer-specific tuning must be counted. It can be introduced within Composer 2 while retaining W → W1 → W2. Its reconstruction target then remains W1, with final quality also measured against W.

**First test / reject if.** Compare the existing superpose reconstruction, a real two-stream encoding, and additive residual coding at equal bytes. Reject if the second code carries little conditional information or its extra reads erase the benefit. Successive refinement is not universally optimal for every source and distortion measure.

**10. Repair the low-dimensional part of the error — perturbation theory and model reduction**

**Principle.** A large error matrix may have a small effective rank. If E = W − Ŵ is concentrated in a few singular directions, an explicit correction UVᵀ can remove much of that error with O(r(m+n)) numbers rather than O(mn).

**Atom.** An optional low-rank correction sidecar, preferably fitted to `EX` on calibration inputs. The decoder can materialize Ŵ + UVᵀ; a future runtime could compute Ŵx + U(Vᵀx). LoftQ provides a related precedent combining quantization with low-rank approximation, in a setting designed for subsequent LoRA fine-tuning. [LoftQ](https://arxiv.org/abs/2310.08659)

**Status and cost.** New. With f16 factors the nominal factor payload is `2r(m+n)` bytes, plus headers/scales; runtime use adds multiplies and memory traffic. Matrix dimension alone does not establish low-rank error.

**First test / reject if.** Measure the singular-value decay of activation-weighted residuals. Sweep very small ranks against using those same bytes for more quantization bits or sparse corrections. Reject if the residual spectrum is flat or the advantage disappears after factor quantization.

**11. Preserve influential exceptions — unequal error protection and robust estimation**

**Principle.** Magnitude and consequence are different quantities. Under a local quadratic loss approximation, a useful saliency proxy is `hᵢ · (wᵢ − ŵᵢ)²`. A modest weight on a frequently used, sensitive feature may deserve more protection than a large but redundant weight.

**Atom.** Extend EXTRACT from fixed top-K magnitude to a budgeted exception selector using *distortion removed per stored byte*. Store a position and correction only when it outperforms the best dense alternative. SqueezeLLM explicitly combines sensitivity-aware nonuniform quantization with sparse storage of outliers and sensitive values. [SqueezeLLM](https://arxiv.org/abs/2306.07629)

**Status and cost.** Extension. Index coding matters greatly: an index inside a 32-weight block has a different cost from a global u32 index. A full mask, packed local indices, and sparse runs have different break-even points.

**First test / reject if.** At equal total bytes, compare magnitude top-K, sensitivity top-K, and whole-block promotion. Reject if gathers or position metadata cost more than the saved dense precision, or if sensitivity does not transfer to held-out prompts.

**12. Fit the correction to the loss — least squares and moment matching**

**Principle.** Restoring a norm is one constraint, not the general minimum-error correction. For a decoded row r, reference row w, and input second moment Cx, the best scalar gain under linear output MSE is `a* = (r Cx wᵀ)/(r Cx rᵀ)` when the denominator is nonzero. This differs from ||w||/||r||. A bias correction to the *output* can target E[(W − Ŵ)x] when the operator supports it.

**Atom.** A POST candidate that fits a small gain, or an explicitly supported gain-plus-bias, against calibration outputs. Store it and apply it at the same position in the decoded pipeline that was validated. A per-weight row offset and a layer output bias are not interchangeable.

**Status and cost.** Extension of `PreserveRowNorm`. One or two values per row plus decoder work. Distinguish zero rows and bound extreme gains. Where scaling can be folded into existing metadata, demonstrate that the fold is exact for the chosen stage ordering.

**First test / reject if.** Compare no correction, norm restoration, least-squares gain, and supported bias correction after all inverse transforms and outlier restoration. Reject if gains improve cosine but worsen held-out output loss, or if tiny side-data costs erase benefits for small tensors.

**13. Let topology determine locality — graph theory and signal processing**

**Principle.** A low-pass filter is useful only when its neighborhood describes meaningful signal structure. Adjacent array elements are not automatically adjacent features. A function-preserving channel permutation can change the behavior of a flat five-tap filter while leaving the underlying unquantized model unchanged.

**Atom.** First add a topology test around repair. Restrict filters to actual row/operator boundaries. As a more exploratory POST, construct a small graph from measured feature relationships and test regularization such as `||v − decoded||² + λvᵀLv`, where L is a graph Laplacian. Graph signal processing supplies the machinery for filtering on non-grid domains. [Signal processing on graphs](https://arxiv.org/abs/1211.0053)

**Status and cost.** New locality criterion; speculative graph repair. A graph learned from original weights or activations must be stored, shared, or reconstructible. Correlated input channels do not automatically imply smooth weight rows.

**First test / reject if.** Compare original ordering, function-preserving permutations, row-bounded repair, and shuffled graphs. Reject a graph prior if its benefits vanish under held-out data or matched metadata cost. A null repair is a valid winner.

**14. Decode a quantization cell using a learned prior — Bayesian estimation and inverse problems**

**Principle.** The minimum mean-square estimator from an observation is its conditional mean. For code k and declared side state m, that is `ŵ = E[w | k,m]`. A quantization cell may have a skewed density, so its midpoint need not be the best reconstruction.

**Atom.** A decoder table fitted per tensor family, shared group, or code/scale context, with carefully limited neighborhood dependence. This differs from per-block Lloyd–Max encoder fitting: the experiment specifically models the decoder's available information and quantization intervals. Under a weight-reconstruction objective, interval-consistent estimates also provide a check on repairs that invent values outside the represented cell.

**Status and cost.** Extension and exploratory application of a standard estimation identity. Pay for prior parameters, table selection, and lookups. Conditional-mean optimality assumes the correct distribution and squared-error loss; it does not imply optimal downstream token loss.

**First test / reject if.** Fit on one set of tensors, validate on separate tensors and activation traces. Compare midpoint, codebook centroids, and posterior estimates. Reject if the prior adds bias under distribution shift or needs original weights during decode.

**15. Trade correlated bias for controlled variance — probability and numerical analysis**

**Principle.** Rounding can be unbiased in expectation. Between grid points a and b, rounding upward with probability (w−a)/(b−a) gives E[Q(w)] = w, absent clipping. A one-time sampled model still has error; expectation over many possible encodings is not a guarantee for that model. Stochastic rounding literature studies exactly this bias–variance tradeoff. [Improved stochastic rounding](https://arxiv.org/abs/2006.00489)

**Atom.** An encode-time stochastic-rounding candidate with reproducible seeds. Test paired or constrained rounding choices that control row-level accumulated bias. A subtractive-dither variant additionally needs a specified decoder-side dither stream and overload handling.

**Status and cost.** New rounding option, related to noise shaping. Sampling at encode time can leave the ordinary deterministic packed decoder unchanged. Searching many seeds adds a model-selection problem and can destroy the unconditioned unbiasedness argument for the selected winner.

**First test / reject if.** Report the distribution of outcomes over seeds, not only the best. Compare rounding bias, held-out task loss, and worst-case regressions. Reject if added variance is more damaging than the systematic error removed.

**16. Charge every explanation bit — entropy coding and minimum description length**

**Principle.** A representation's rate includes the model of its codes, not just the code symbols. If symbol frequencies are skewed, entropy coding can reduce storage losslessly; if every tiny block has its own complex model, the model descriptions may cost more than they save.

**Atom.** A serialization layer that can entropy-code symbols, strategy maps, scale deltas, and exception positions in independently restartable chunks. Share small codebooks where doing so reduces combined payload and metadata. Neural-network weight sharing followed by Huffman coding has a longstanding precedent. [Deep Compression](https://arxiv.org/abs/1510.00149)

**Status and cost.** New container candidate, with a distinction between compact storage and the runtime representation. Codes alone have a ternary alphabet in current symmetric Q2; their ideal entropy can be below two bits, but the block scale and headers remain. Halving a 4-byte scale to 2 bytes saves 0.5 bits/weight at BLOCK=32: symmetric Q2 moves from roughly 3 to 2.5 bits/weight before headers, a 16.7% reduction, not 50%.

**First test / reject if.** Encode real streams and measure file size, seek latency, startup decode, and working-set memory. Reject schemes whose model tables or decode work cost more than the compression is worth for the intended use.

**17. Refine only where error warrants it — adaptive meshes and multiscale approximation**

**Principle.** A fixed resolution wastes resources on easy regions and underserves difficult ones. Subdivide a group only when the reduced distortion is worth the extra scale, split description, and execution cost.

**Atom.** A dyadic block partition with a bounded set of group sizes, selected by dynamic programming over measured rate–distortion costs. This extends the existing fixed nested-scale atom into adaptive granularity. It need not move tensor elements or change feature identity.

**Status and cost.** New adaptive policy; related to the fixed `ComposeNestedScale`. Pay for split flags and all leaf metadata. A runtime may need to restrict legal partitions to preserve contiguous vectorized access.

**First test / reject if.** Compare fixed 32-weight blocks with bounded 8/16/32/64 group choices at matched total bytes. Include the rounding error of stored scales. Reject if most groups split to the finest size, or if irregular traversal overwhelms the fidelity gain.

**18. Preserve dynamics over time — control theory and stability analysis**

**Principle.** A small parameter change can alter a recurrent system's memory time constant. In the scalar example hₜ₊₁ = a hₜ + uₜ, the steady response to constant input scales as 1/(1−a), and its sensitivity as 1/(1−a)². Sensitivity rises sharply near a = 1. For matrices, long-horizon products and nonnormal transient growth matter too; eigenvalues alone are insufficient.

**Atom.** A VERIFY policy for actual recurrent/state-space operators using state trajectories, impulse responses, and long-context error. Store explicit operator identity and parameterization; do not infer stability from a tensor's spelling or a histogram alone. QMamba demonstrates that state-space models can have temporally varying states and unusual parameter distributions, although its experiments are on vision models. [QMamba](https://arxiv.org/abs/2501.13624)

**Status and cost.** New, architecture-dependent test and precision allocator. The project's historical `ssm_a` failures make this worth investigating, but the correct transition must be derived from the target model's actual operation.

**First test / reject if.** Compare trajectory error and loss at increasing context lengths, including the deployed recurrence. Reject a proposed stability proxy if it cannot predict functional degradation. Preserve precision when a tiny tensor controls expensive long-horizon behavior.

**19. Protect fragile decisions and coupled operators — robust geometry**

**Principle.** A discrete choice can change abruptly despite small average error. If the gap between the largest and second-largest logits is m, an infinity-norm perturbation below m/2 preserves the argmax. This is only an argmax statement, not a guarantee of probabilistic or language-model quality.

**Atom.** Operator-specific VERIFY atoms for attention scores, routing top-K boundaries, gating, and normalization. Quantize coupled Q/K or gate/up/down operators as a measured block when their error interactions matter. For attention, the perturbation contains `δQ Kᵀ + Q δKᵀ + δQ δKᵀ`; independent per-tensor cosine does not measure these terms.

**Status and cost.** New. Requires real operator traces and tensor-role mapping, with held-out prompts covering small-margin decisions. Margins can identify where a bit upgrade is valuable rather than making every tensor uniformly more precise.

**First test / reject if.** Record margin distributions, routing changes where applicable, and downstream loss together. Reject a margin gate if it overprotects harmless changes or fails to predict task degradation. Continuous outputs and nonlinearities still need direct error checks.

**20. Treat candidate selection as an experiment — statistics and decision science**

**Principle.** Selecting the best of many noisy measurements creates optimism. Four fixed probes reused across a large search are a screening set, not independent evidence that the selected stack generalizes. More probes of the wrong distribution also do not fix the problem in principle 1.

**Atom.** Separate calibration, candidate selection, and final held-out evaluation. Increase sample count for close decisions and stop early for clearly poor candidates. Use paired comparisons because candidates see the same examples. Valid sequential confidence methods exist, subject to assumptions about the samples and losses. [Confidence sequences](https://arxiv.org/abs/1810.08240)

**Status and cost.** New experimental policy. Use document-level or prompt-level sampling units when adjacent tokens are correlated. An ordinary confidence interval repeatedly inspected during adaptive stopping does not automatically retain its nominal coverage. Examine high-loss prompt groups and tails as well as the mean.

**First test / reject if.** Re-select stacks on several independent calibration/selection splits, then measure held-out stability and failure rates. Reject fragile winners. A distribution-shift stress set should remain separate from the data used to tune it.

**21. Encode predictions plus innovations — neuroscience and predictive compression**

**Principle.** If declared shared structure predicts part of a signal, encode the remaining innovation. Predictive coding models of visual cortex explicitly transmit prediction residuals between hierarchical levels. That is a source of an algorithmic hypothesis here, not evidence that model weights share cortical statistics. [Rao and Ballard](https://www.nature.com/articles/nn0199_79)

**Atom.** As an exploratory PRE/EXTRACT pair, test shared templates or dictionaries across truly comparable tensors, then quantize their residuals. Every predictor must be built from decoder-available anchors or a stored dictionary. Expert tensors with the same operator role are one possible experiment; unrelated rows are not automatically aligned.

**Status and cost.** New and speculative. Count anchors, coefficients, residual payloads, alignment/permutation state, and dependency metadata. Decode anchors first and use their decoded values when fitting dependent residuals to avoid drift. Avoid long dependency chains that compromise random access.

**First test / reject if.** Compare residual entropy and activation error against independent coding, using random tensor pairings as controls. Reject if shared structure disappears after accounting for alignment and side data, or if it raises runtime memory substantially.

**22. Optimize the bottleneck the machine experiences — computer architecture and energy cost**

**Principle.** A smaller file does not imply faster or smaller resident inference. A useful lower-bound model is `T ≥ max(bytes_moved/bandwidth, operations/throughput)`, with launch, synchronization, and transfer costs considered separately. Roofline formalizes this interaction. [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)

**Atom.** A hardware cost model attached to candidate records, and eventually a tile decoder that can feed a matrix product without materializing an entire f32 tensor. Shared lookup tables, aligned packing, and local exception streams are concrete candidates. FLUTE provides a direct example of designing lookup-quantized layouts and fused dequantization/matmul together. [FLUTE](https://arxiv.org/abs/2407.10960)

**Status and cost.** New deployment boundary. The current CUDA module accelerates fidelity matmul; it does not implement compressed model inference. For a storage-only goal, measure bytes and load time. For an inference goal, additionally measure resident memory, tokens/second, latency, and, if available, energy/token on the actual runtime.

**First test / reject if.** Benchmark complete candidate decode plus matmul across representative batch/context sizes, including metadata and host/device transfers. Reject candidates that win in nominal bits but lose on the target workload's measured bottleneck.

**How these principles should compose**

Each proposed atom should declare its input domain, output domain, shape requirements, original-data requirements, stored side state, deterministic decoder behavior, byte cost, and runtime cost. A valid pipeline is an executable dependency graph, not just a list of names. Removing a row mean is an affine translation; an orthogonal rotation is linear. Their invariants differ. Extraction, filtering, inverse transforms, and row scaling generally do not commute.

The first useful composition experiment is:

`calibration → [optional precondition] → existing transform candidate → [optional sensitivity exceptions] → [one quantizer/rounding family] → [optional stored correction] → serialize → decode from stored state → verify real outputs`.

Treat vector codebooks, trellis coding, and additive coding initially as alternative QUANTIZE families. Treat low-rank, sparse, and row-affine corrections initially as alternative uses of a common correction budget. Add combinations only after their interactions and complete costs have been measured. In particular, fitting a whole-row norm to the remaining bulk before restoring a large outlier double-counts energy.

For the existing sequential W → W1 → W2 design, measure E1 = W1−W and E2 = W2−W1 as well as the total. In an input-weighted quadratic metric:

`Dtotal = D1 + D2 + 2 tr(E1 Cx E2ᵀ)`.

The cross term can improve or harm the total. A deterministic second stage cannot recover information absent from W1 without additional side information or a prior; it can still yield a better size/error tradeoff or denoise under justified assumptions. The companion's approximate first stage should be labeled in experiments, and comparisons should not silently treat it as the exact main picker.

**Recommended experiment order**

| Order | Experiment | Why it belongs here |
|---|---|---|
| 0 | Reproduce accepted reconstructions from serialized records; account for all side state; distinguish failed gates. | Establishes what every later result actually measures. |
| 1 | Real-activation output error, calibrated gain correction, optimized Q2 scale/alphabet, and a budgeted tensor selector. | Closest to current code; can reveal gains before introducing an entirely new representation. |
| 2 | Compare correlation-directed feedback and joint rounding; then sensitivity-based exceptions. | Exploits task geometry at a fixed or tightly controlled storage cost. |
| 3 | Run separate scalar, lattice, trellis, and additive coding experiments. | Direct tests of whether a richer code space improves the Q2 frontier. |
| 4 | Compare low-rank correction, entropy coding, and adaptive block sizes. | Determines whether residual structure or metadata dominates the remaining cost. |
| 5 | Try topology-aware repair, Bayesian decode, predictive templates, and architecture-specific temporal/margin gates. | Broader hypotheses with more assumptions or operator integration work. |
| Deployment decision | Measure tile decode and actual runtime behavior for finalists. | Determines whether an encoding's numerical gain transfers to the intended use. |

Use the exact source model hash and a supported source dtype. Compare full artifact bytes, including needed model metadata, rather than treating dense f32 as the only denominator. Calibrate on representative text/code/math and, when relevant to the model, other supported modalities. Keep final evaluation prompts disjoint. Report held-out negative log likelihood/perplexity, relevant downstream tasks, raw/operator output error, Q2/Q4/Q8 allocation by parameters as well as tensor counts, gate failures, metadata bytes, and decode cost. Compare the current codec with appropriate established quantizers at matched size and runtime constraints; do not infer an advantage from synthetic tensors.

A whole-model inference check should consume the reconstructed artifact after real serialization and dtype conversion. Coherent chat output is a useful smoke test, but it does not replace held-out loss or representative task evaluation. Repeat across at least a small model and a larger or architecturally different model before treating any winner as a general improvement.

**Reproducing the study diagnostics**

From the repository root, the commands below compile only the normal CPU codec and a standalone diagnostic. The diagnostic writes its fixtures into `target/first-principles-2026-09-05`; it never edits production Rust source. The first test attempt with `--offline` failed because the optional `cudarc` dependency was absent from the local index. The normal test command resolved the index and passed; the subsequent offline build succeeded.

```bash
cargo test --workspace --all-targets
cargo build --offline -p atom-quantizer
rustc --edition=2021 -O research/first-principles-2026-09-05/probe.rs -o target/first-principles-probe
target/first-principles-probe target/debug/atom-quantizer target/first-principles-2026-09-05
```

Environment observed: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, `cargo 1.97.1 (c980f4866 2026-06-30)`, Linux CPU path. The source and results are [probe.rs](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/research/first-principles-2026-09-05/probe.rs), [evidence.json](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/research/first-principles-2026-09-05/evidence.json), and [validation.txt](/home/jesse/Desktop/Projects/atom-quantizer/atom-quantizer/research/first-principles-2026-09-05/validation.txt). Published precedents above are motivation for future experiments; none of the 22 proposed improvements was implemented or model-benchmarked during this study.
