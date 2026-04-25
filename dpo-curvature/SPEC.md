# Implicit-Reward Curvature Predicts DPO Overoptimization
**Implementation Specification — v0.1**
**Author:** Rishav (anubrat23@gmail.com)
**Target:** single-GPU 1×A100-80G, ~10 wall-clock hours total
**Status:** spec frozen for implementation; deviations require a written addendum.

---

## 0. TL;DR

Train Pythia-410M on UltraFeedback preferences with DPO. At each saved checkpoint, measure (a) a token-swap curvature `C` of the implicit reward and (b) the overoptimization gap `Δ` between the implicit reward and an external gold RM. Test whether `C` predicts `Δ`. Then train a second policy with `λ·C` added to the DPO loss, and test whether it reduces `Δ` at matched preference accuracy. Single-GPU, ~10 A100-hours, no distributed training.

---

## 1. Hypotheses

**H1 (correlational).** Across DPO checkpoints `{θ_t}`, the probe-set curvature `C̄(θ_t)` correlates positively (Spearman `ρ > 0.5`) with the overoptimization gap `Δ(θ_t)` on a held-out evaluation set.

**H2 (interventional).** A curvature-regularized objective `L = L_DPO + λ·L_curv` produces, at matched preference accuracy on a held-out split, a strictly smaller `Δ` than vanilla DPO for at least one `λ` in the swept range.

**Pre-registered null outcomes** (each independently publishable):

- **N1.** `C̄` increases monotonically while `Δ` does not — falsifies the flat-minima → overopt link in the DPO setting.
- **N2.** `C̄` and `Δ` correlate, but the regularizer fails to improve the (pref-acc, Δ) Pareto frontier — `C` is a symptom, not a cause.
- **N3.** Probe-set sensitivity dominates: `ρ` flips sign across swap distributions — diagnostic is non-robust.

---

## 2. System Components

| Symbol | Meaning |
|---|---|
| `π_ref` | SFT-initialized Pythia-410M, frozen during DPO. |
| `π_θ` | Trainable Pythia-410M, initialized from `π_ref`. |
| `r_θ(x,y) := β · [log π_θ(y\|x) − log π_ref(y\|x)]` | Implicit reward induced by DPO. |
| `r_gold(x,y)` | External reward model, frozen. See §3.4. |
| `P` | Probe set: 512 fixed `(x, y_chosen)` pairs from a held-out shard. |
| `E_pref` | Held-out preference eval: 2K pairs. |
| `E_gen` | Generation eval: 1K prompts. |

---

## 3. Data

### 3.1 Source
`HuggingFaceH4/ultrafeedback_binarized`, splits `train_prefs` / `test_prefs`.

### 3.2 Splits (deterministic, seeded `numpy.random.default_rng(20260425)`)

| Split | Size | Source |
|---|---|---|
| SFT-train (chosen-only) | 8 000 | shard A of `train_prefs` |
| DPO-train (triples) | 16 000 | shard B of `train_prefs` |
| Probe set `P` | 512 | shard C of `train_prefs` (disjoint from A, B) |
| Pref eval `E_pref` | 2 000 | `test_prefs` |
| Gen eval `E_gen` | 1 000 | `test_prefs` |

### 3.3 Preprocessing
- Tokenizer: `EleutherAI/pythia-410m-deduped`.
- Template: `"PROMPT:\n{x}\n\nRESPONSE:\n{y}"` (Pythia is not chat-tuned; trivial template avoids assumptions).
- Max sequence length: 512. Drop pairs whose chosen or rejected response exceeds 256 tokens.
- BOS prepended; EOS appended to the response.
- Loss masking: prompt tokens masked from CE loss in SFT; in DPO, log-probs are summed over response tokens only.

### 3.4 Gold reward model

**Default:** `OpenAssistant/reward-model-deberta-v3-large-v2`. Externally trained → avoids circularity of training the gold on the same UltraFeedback split as the proxy.

**Fallback:** Pythia-1B trained on full `train_prefs`, 2 epochs (~1.5 hr). Use only if external RM is unavailable; document the data overlap as a limitation.

---

## 4. Models and Training

### 4.1 `π_ref` — SFT
| Field | Value |
|---|---|
| Init | `EleutherAI/pythia-410m-deduped` |
| Loss | causal LM on response tokens only |
| Optimizer | AdamW, betas (0.9, 0.95), wd 0 |
| LR | 1e-5 peak, 50-step warmup, cosine to 1e-6 |
| Effective batch | 32 (16 micro × 2 grad-accum), seq 512 |
| Steps | 1 epoch over SFT-train (~250 steps) |
| Precision | bf16 weights, fp32 master in optimizer |
| Wall time | ≈30 min |
| Output | `runs/sft/` |

### 4.2 `π_θ` — DPO baseline
| Field | Value |
|---|---|
| Init | `π_ref` weights (also frozen reference for log-ratios) |
| Loss | `L_DPO = −E[log σ(β · (Δ_w − Δ_l))]`, `Δ_y = log π_θ(y) − log π_ref(y)` |
| `β` | 0.1 |
| Optimizer | AdamW |
| LR | 5e-7 peak, 50-step warmup, cosine to 5e-8 |
| Effective batch | 16 (8 micro × 2 grad-accum), seq 512 |
| Steps | 2 epochs over DPO-train (~4 000 steps) |
| Save cadence | every 250 steps → 16 checkpoints |
| Wall time | ≈1.5 hr |

### 4.3 `π_θ` — curvature-regularized DPO
Identical to 4.2 plus `L = L_DPO + λ · L_curv`. Sweep `λ ∈ {0.01, 0.1, 1.0}`. See §5.2 for `L_curv`.

---

## 5. Curvature Estimator

### 5.1 Probe-set token-swap curvature (offline, for analysis)
For `(x, y) ∈ P` with response length `L`:

1. Choose `M = 16` positions uniformly without replacement (or all positions if `L < 16`).
2. For each position `i`, sample `K = 8` replacement tokens `y'_i,k ~ Q` (swap distribution).
3. Define `s_θ(x, y, i, k) = r_θ(x, swap(y, i, k)) − r_θ(x, y)`.
4. Per-position curvature: `c_i = (1/K) Σ_k s_θ(x, y, i, k)²`.
5. Per-pair curvature: `C(x, y) = (1/M) Σ_i c_i`.
6. Probe-set curvature: `C̄(θ) = mean over P of C(x, y)`.

**Swap distribution `Q` (ablation knob).** Three variants:

| Name | Definition |
|---|---|
| `Q_topk` | uniform over `π_ref`'s top-50 next-token distribution at position `i` |
| `Q_unif` | uniform over the full 50K-token vocabulary |
| `Q_rand-id` | uniform over a fixed random subset of 1024 in-vocab tokens |

**Default reported in main figures:** `Q_topk`. Others go in the appendix.

**Justification.** Under a Lipschitz assumption on `r_θ` in token-embedding space and `Q` centered around `y_i`, `C` is a Monte-Carlo estimator of `E_q[(r(y) − r(y+q))²] ≈ qᵀ H q + O(‖q‖³)`, i.e. a directional second-moment of the local Hessian. We do not claim formal flat-minima sharpness — `C` is a cheap, well-defined, model-agnostic curvature proxy.

### 5.2 In-loop curvature (during regularized training)
On each step, for the chosen completion `y_w` of each minibatch element:

- Sample `M_train = 2` positions, `K_train = 2` swaps per position from `Q_topk`.
- `L_curv = mean over (i, k) of s_θ(x, y_w, i, k)²`.
- Backprop `λ · L_curv` jointly with `L_DPO`.
- Cost: `2 · (2 + M_train·K_train) = 12` forwards per minibatch element vs. 4 for vanilla DPO. Roughly 2.5× wall-time per step.

---

## 6. Overoptimization Gap

For each checkpoint `θ_t`:

1. Sample 4 completions per prompt from `π_θ` on `E_gen` (1K prompts), `T=0.9`, `top_p=0.95`, `max_new=256`. Total: 4 000 generations.
2. For each `(x, ŷ)`, compute `r_θ(x, ŷ)` and `r_gold(x, ŷ)`.
3. **`Δ_raw(θ_t)`** = `mean[r_θ] − mean[r_gold]` after per-axis z-normalization across the run's checkpoints.
4. **`Δ_BoN(θ_t, n)`**: for `n ∈ {1, 2, 4}`, `mean over prompts of max over n samples of r_gold`. Plot vs `KL(π_θ ‖ π_ref)`. Overopt = gold ↓ while implicit ↑.
5. **`Pref-acc(θ_t)`**: fraction of `E_pref` pairs where `r_θ(y_w) > r_θ(y_l)`.

Primary scalar `Δ` for the H1 correlation: `Δ_raw` after z-normalization within run.

---

## 7. Experiments

### 7.1 Run inventory (10 A100-hr budget)

| # | Run | Wall time | Checkpoints |
|---|---|---|---|
| 0 | SFT (`π_ref`) | 0.5 hr | 1 |
| 1 | DPO baseline, `β=0.1` | 1.5 hr | 16 |
| 2 | DPO + curv-reg, `λ=0.01` | 1.5 hr | 16 |
| 3 | DPO + curv-reg, `λ=0.1` | 1.5 hr | 16 |
| 4 | DPO + curv-reg, `λ=1.0` | 1.5 hr | 16 |
| 5 | Probe + overopt eval (all checkpoints) | 2.5 hr | — |
| | **Total** | **9.0 hr** | |

Buffer: 1 hr for re-runs / debugging.

### 7.2 Headline analyses

- **A1.** `C̄(θ_t)` trajectory across training. Reported per `Q`. Expectation: monotone increasing.
- **A2.** Spearman `ρ(C̄, Δ_raw)` per run and pooled. Bootstrap CI (n=1000) over checkpoints.
- **A3.** Pareto frontier: `pref-acc` vs `Δ_raw` per (run, checkpoint). Compare baseline trajectory to curv-reg trajectories.
- **A4.** Goodhart curve: `r_gold` vs `r_θ` per checkpoint. Slope < 1 ⇒ overopt regime.

### 7.3 Ablations / sanity checks

- **B1.** Swap distribution: `ρ` for `Q_unif`, `Q_topk`, `Q_rand-id`.
- **B2.** Probe-size convergence: `ρ` stability with `|P| ∈ {64, 128, 256, 512}`.
- **B3.** Position sampling: uniform vs. response-end-weighted.
- **B4.** Seed variance: baseline DPO repeated at 3 seeds (only if buffer allows).

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `C` dominated by tokenizer rare-token artifacts | Default `Q_topk`; report all three. |
| `Δ` noisy at 1K eval prompts | Bootstrap CI; per-prompt deltas; 4 completions/prompt. |
| Pref-acc collapses before overopt visible | Save every 250 steps; eval from step 250. |
| External gold RM domain mismatch | Report rank correlation; cross-validate with self-trained gold on a 25 % shard. |
| Curv-reg destroys pref-acc at all `λ` | Sweep includes `λ=0.01`; if all degrade pref-acc by >5 pp, report as N2. |
| Curv-reg gradient instability | Clip per-token swap-Δ magnitude at ±5; gradient clip at 1.0 globally. |

---

## 9. Repository Layout

```
dpo-curvature/
├── SPEC.md                         # this file
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── sft.yaml                    # stage 2
│   ├── dpo_baseline.yaml           # stage 3
│   ├── dpo_curv.yaml               # stage 4
│   ├── probe.yaml                  # stage 4
│   └── overopt.yaml                # stage 5
├── src/dpocurv/
│   ├── __init__.py
│   ├── types.py
│   ├── data/                       # stage 1
│   │   ├── __init__.py
│   │   ├── ultrafeedback.py
│   │   ├── splits.py
│   │   └── probe_set.py
│   ├── models/                     # stage 2/3
│   │   ├── __init__.py
│   │   ├── policy.py
│   │   └── reward_model.py
│   ├── training/                   # stage 2/3/4
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── sft_trainer.py
│   │   ├── dpo_trainer.py
│   │   └── curv_dpo_trainer.py
│   ├── eval/                       # stage 5
│   │   ├── __init__.py
│   │   ├── implicit_reward.py
│   │   ├── curvature.py
│   │   ├── generate.py
│   │   ├── score.py
│   │   └── overopt.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── logging.py
│   │   ├── checkpoint.py
│   │   └── io.py
│   └── cli/                        # stage 6
│       ├── __init__.py
│       ├── run_sft.py
│       ├── run_dpo.py
│       ├── run_curv_dpo.py
│       ├── build_probe.py
│       ├── eval_curvature.py
│       └── eval_overopt.py
├── scripts/
│   ├── 00_env_check.sh
│   ├── 01_train_sft.sh
│   ├── 02_train_dpo_baseline.sh
│   ├── 03_train_curv_dpo_sweep.sh
│   ├── 04_eval_all.sh
│   └── 05_make_plots.sh
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py               # stage 0
│   ├── test_data.py                # stage 1
│   ├── test_dpo_loss.py            # stage 3
│   ├── test_curvature.py           # stage 4
│   └── test_overopt.py             # stage 5
└── notebooks/
    └── 01_paper_figures.ipynb
```

---

## 10. Implementation Stages

Each stage ends with: (a) `pip install -e .`; (b) `python -c "import ..."` import-check; (c) targeted `pytest`; (d) explicit go/no-go before advancing.

| Stage | Scope |
|---|---|
| **0** | Conda env, project skeleton, configs scaffold, smoke tests. *(this turn)* |
| **1** | Data pipeline: UltraFeedback loader, splits, probe-set builder. |
| **2** | SFT trainer + Gold RM (external loader; optional self-trained fallback). |
| **3** | DPO trainer (vanilla) + implicit-reward computation. |
| **4** | Curvature estimator (offline) + curv-regularized DPO trainer. |
| **5** | Generation, gold-scoring, overopt analysis (BoN, Goodhart, Δ). |
| **6** | Experiment orchestration: configs, sweep launcher, structured logging. |
| **7** | Plotting, statistical tests, paper-ready figures. |

---

## 11. Reproducibility

- Master seeds in `configs/seeds.yaml`: `[20260425, 20260426, 20260427]`.
- All randomness via `dpocurv.utils.seed.set_seed`.
- Each run dumps `config.yaml` + `git_sha.txt` + `pip_freeze.txt` to `runs/<run_name>/meta/`.
- Versions pinned in `environment.yml` and `pyproject.toml`.
- `torch.use_deterministic_algorithms(True, warn_only=True)` in *evaluation only* (preserves flash-attention during training).

---

## 12. Out of Scope (v0.1)

- Multi-GPU / FSDP / DeepSpeed.
- Models > 1B params.
- Non-Pythia architectures.
- Online RLHF (PPO, GRPO).
- Length-controlled / SimPO-style variants.

These belong in a follow-up. The v0.1 scope is the minimal experiment that can falsify or support H1+H2.
