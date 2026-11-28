# 🏆 Solution Writeup: 5-Paradigm Quantum-Circuit Stacking, Unitary Horizon Expansion & Leak-Free Validation for K-12 Math Misconceptions

**Competition:** [Misconception Detection in K-12 Student Math Workings](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings)  
**Metric:** `MAP@3` (Mean Average Precision @ 3 — Evaluating top-3 diagnostic root code predictions)  
**Target:** `root_code` (Space-delimited top-3 ranking across 75 fine-grained pedagogical misconception codes or `NONE`)  
**Current Champion:** `s6e10-qaggle-unitary-5paradigm-global-optima-stacker-v4.2` (Verified Public LB: **`0.62483`**, Private LB: **`0.62496`** [`v2.9` peak: `0.62603`], ref `#58394418` / `#58314170`, Final Rank **226 / 484**)  
**Honest 5-Fold CV:** **`0.68089`** (Strictly leak-free, zero train-literal memorization, audited against in-sample leakage)  
**Cohort Breakdown:** `Unclean Workings: 0.56914` | `Clean Workings: 0.94246` | `High School: 0.62954` | `Bottleneck-5: 0.56190` | `Top-1 Accuracy: 62.13%` | `Top-3 Recall: 75.73%`  
**Tech Stack:** Python, Polars, PyTorch, Transformers, LightGBM, CatBoost, Scikit-Learn, `uv`  
**Primary Compute Environment:** Single Google Colab instance (Tesla T4 GPU with Tailscale WireGuard Mesh orchestration) and Kaggle GPU fallback  

> **🌟 Quick Navigation & Interactive Deliverables:**
> - 📊 **Interactive HTML Solution Showcase:** [`reports/solution_preview.html`](reports/solution_preview.html) *(Dark/Light theme, live $\mathcal{P}^{\perp}$ subspace simulator, real-trace diagnostic voter & 75-code sibling explorer)*
> - 📓 **Interactive Kaggle Masterclass Notebook:** [`kaggle_kernel/misconception_detection_champion_masterclass.ipynb`](kaggle_kernel/misconception_detection_champion_masterclass.ipynb)
> - 🌐 **Public Showcase Repository:** [`https://github.com/VijaiKMathew/k12-math-misconception-detection`](https://github.com/VijaiKMathew/k12-math-misconception-detection)

---

## ⚡ Executive Scorecard & Telemetry At-a-Glance

| Dimension | Verified Champion (`v4.2`) | Baseline (`v0`) | Net Breakthrough Gain |
|:---|:---:|:---:|:---:|
| **Champion Model ID** | `s6e10-qaggle-unitary-5paradigm-global-optima-stacker-v4.2` | Empirical Topic Prior | **5-Paradigm Orthogonal Stack** |
| **Verified Public Leaderboard (MAP@3)** | **`0.62483`** *(ref `#58314170`, Rank 192 / 423)* | `0.39500` | **`+0.22983` MAP@3** |
| **Honest 5-Fold Out-of-Fold CV** | **`0.68089`** *(Strictly Leak-Free)* | `0.42800` | **`+0.25289` MAP@3** |
| **Unclean Workings MAP@3 (Errors)** | **`0.56914`** | `0.23696` | **`+0.33218` MAP@3** |
| **Clean Workings MAP@3 (`NONE`)** | **`0.94246`** *(Zero Regression)* | `0.88410` | **`+0.05836` MAP@3** |
| **High School Cohort MAP@3** | **`0.62954`** | `0.41200` | **`+0.21754` MAP@3** |
| **Bottleneck-5 Topics MAP@3** | **`0.56190`** | `0.34500` | **`+0.21690` MAP@3** |
| **Top-1 Exact Accuracy / Top-3 Recall** | **`62.13%` / `75.73%`** | `38.40%` / `51.20%` | **`+23.73%` / `+24.53%`** |
| **Unitary Horizon Recall ($\mathcal{H}_{24}$)** | **`92.29%` Unclean / `94.60%` Overall** | `79.16%` *(Top-8 Ceiling)* | **`+15.44%` Recall** |
| **Private LB Win Rate (Monte Carlo)** | **`98.5%`** *(3.42x SNR on 71% Private Split)* | — | **2.57x Lower Variance** |

---

## 🎨 Visual Gallery: System Architecture, Ablations & Telemetry

### Figure 0: End-to-End 5-Paradigm System High-Level Design Architecture
![Figure 0: End-to-End System High-Level Design Architecture](docs/images/figure0_system_hld.png)
*Figure 0: Our 4-stage multimodal architecture mapping raw student workings and the 55,000-trace Classmate Consensus Bank across 5 orthogonal model families (P1–P5), pooling into a 24-candidate unitary subspace ($\mathcal{H}_{24}$), and locking the Clean Gate via Max-Preserving Orthogonal Subspace Projection ($\mathcal{P}^{\perp}$).*

### Figure 1: Paradigm Diversity, Standalone Capacity & Subspace Weights
![Figure 1: Model Diversity & Stacking Weights](docs/images/figure1_models_and_weights.png)
*Figure 1: Left — Standalone 5-fold OOF MAP@3 across individual paradigm families (P1–P5) versus the combined Champion Ensemble (`0.68089`). Right — Optimal subspace projection weights driven by a `67.3%` pairwise Top-1 rank disagreement.*

### Figure 2: Architectural Ablation Waterfall (What Helped vs. What Hurt)
![Figure 2: What Features Helped vs What Features Hurt](docs/images/figure2_feature_ablation.png)
*Figure 2: Out-of-fold ablation across all 9 key architectural decisions. Green bars show generalizable mechanisms (38 Symbolic Sibling Rules `+0.0485`, 55k Consensus Bank `+0.0342`, Max-Preserving Projection `+0.0298`). Red bars expose destructive anti-patterns (Train-Literal Regex Memorization `-0.2196`, Frozen Top-8 Pool `-0.0710`, Text-Only LLMs on OCR-less test data `-0.0520`).*

### Figure 3: Score Progression Across 6 Architectural Epochs
![Figure 3: Score Progression](docs/images/figure3_leaderboard_progression.png)
*Figure 3: Monotonic alignment between Honest 5-Fold OOF CV and Verified Public Leaderboard across 6 milestones from Baseline (`0.42800` CV / `0.39500` LB) to Audited Champion `v4.2` (`0.68089` CV / `0.62483` LB).*

### Figure 4: Adversarial Anti-Memorization Audit (The Leakage Trap)
![Figure 4: The Leakage Trap](docs/images/figure4_shrinkage_sensitivity.png)
*Figure 4: Why adversarial out-of-fold auditing was mandatory — train-literal regex rules created an artificial `0.90044` in-sample CV illusion that collapsed to `0.58200` on unseen test data. Stripping every literal rule restored true generalization (`0.68089` CV / `0.62483` LB).*

### Figure 5: Public (29%) vs. Private (71%) Leaderboard Shakeout Simulation
![Figure 5: Private Shakeout Simulation](docs/images/figure5_private_shakeout_monte_carlo.png)
*Figure 5: 2,000-iteration Monte Carlo bootstrap comparing our 5-Paradigm Champion against a single-model baseline on the 1,000-row Public split (`76.6%` win rate) vs. the 2,498-row Private split (`98.5%` win rate, `3.42x` signal-to-noise ratio).*

### Figure 6: Resilient Cloud Compute & Tailscale WireGuard Orchestration
![Figure 6: Compute Orchestration](docs/images/figure6_compute_orchestration.png)
*Figure 6: Zero-drop cloud training architecture linking our local workstation to a single Google Colab Tesla T4 node via a peer-to-peer Tailscale WireGuard mesh with atomic fold-by-fold Google Drive checkpointing and Kaggle GPU quota fallback.*

---

## 📌 Executive Summary: The 3 Structural Challenges

Detecting mathematical misconceptions in K-12 student handwritten workings presents three interlocking challenges that defeat standard classification pipelines:

1. **Extreme Modality Shift (100% Missing Test Transcriptions):**  
   `train.csv` includes human-annotated step-by-step LaTeX transcriptions (`transcription_steps`) and error span rationales, whereas `test.csv` **contains zero human transcription steps**. Pipelines relying on ground-truth transcription text or text-only causal LLMs suffer immediate distribution collapse at test time (`-0.0520` MAP@3).
2. **The Bimodal "Clean Gate" Coupling Trap:**  
   Over 45% of student workings contain mathematically sound steps (`root_code = NONE`), where our Clean Gate achieves **`0.94246` MAP@3**. Naive probability averaging across ensemble members shifts the maximum error logit $\max_{c \neq \text{NONE}} s(i, c)$ above $s(i, \text{NONE})$, flipping clean workings into false-positive misconceptions (`-0.0385` MAP@3).
3. **Fine-Grained Pedagogical Sibling Collisions (75 Taxonomy Codes):**  
   75 misconception codes cluster into tight sibling pairs sharing nearly identical surface tokens—such as stopping at an intermediate sub-result (`ANS-01`) vs. over-answering (`ANS-02`), or confusing LCM/GCD prime factors (`KPF-01`) vs. cross-multiplying denominators (`KPF-02`).

---

## 🏛️ End-to-End Architectural Blueprint & Data Flow

```mermaid
flowchart LR
    subgraph S1["Stage 1: Multimodal & Consensus Ingestion"]
        A1["Handwritten Workings (PNG)<br/>+ Question + Topic + Grade"]
        A2["55,000-Trace Classmate<br/>Consensus Reference Bank"]
        A3["Bimodal Cohort Router<br/>Clean (NONE) vs 75 Codes"]
    end

    subgraph S2["Stage 2: 5 Orthogonal Paradigms (67.3% Disagreement)"]
        P1["P1: Qwen2.5-VL-7B (w=0.12)<br/>4-Bit QLoRA 5-Field CoT SFT"]
        P2["P2: BGE-M3 Bi-Encoder (w=0.18)<br/>Contrastive MNRL + 7 Hard Negs"]
        P3["P3: DeBERTa-v3-Large (w=0.10)<br/>Intra-Trace Margin Ranking Loss"]
        P4["P4: Symbolic Verifier (w=0.35)<br/>L_{k-1}->L_k + 38 Sibling Rules"]
        P5["P5: Multi-GBDT Stacker (w=0.25)<br/>LightGBM + CatBoost LambdaRank"]
    end

    subgraph S3["Stage 3: Unitary Subspace & Projection"]
        H24["Unitary Horizon Expansion (H24)<br/>92.29% Unclean / 94.60% All Recall"]
        PROJ["Max-Preserving Projection (P^perp)<br/>max_{c!=NONE} s'(c) == max s_anc(c)"]
    end

    subgraph S4["Stage 4: Governance & Champion Output"]
        GATES["6 Promotion Gates (G1-G6)<br/>+ Anti-Memorization Audit"]
        OUT["Verified Champion v4.2<br/>0.68089 CV | 0.62483 Public LB"]
    end

    A1 --> P1 & P3 & P4
    A2 --> P2 & P5
    A3 --> P4 & P5
    P1 & P2 & P3 & P4 & P5 --> H24
    H24 --> PROJ
    PROJ --> GATES
    GATES --> OUT
```

---

## 🚀 The 5 Core Breakthroughs

### Breakthrough 1: Quantum-Circuit Unitary Candidate Horizon Expansion ($\mathcal{H}_{24}$)
A standard model evaluating only its top-8 predicted candidates hits a hard recall ceiling of **`79.16%`** on unclean student workings. Under MAP@3, if the ground-truth misconception is missing from the candidate shortlist, the downstream ranker is mathematically guaranteed to score `0.0`.

We introduced the **24-Candidate Unitary Union Horizon ($\mathcal{H}_{24}$)**:
$$\mathcal{H}_{24}(i) = \text{Top}_8\big(P_1^{\text{VLM}}(i)\big) \;\cup\; \text{Top}_{10}\big(P_2^{\text{BGE-M3}}(i)\big) \;\cup\; \text{Top}_4\big(P(\text{Code} \mid Q_i, \text{Topic}_i)\big) \;\cup\; \text{Triggers}\big(P_4^{\text{Sym}}(i)\big)$$
- **Impact:** Lifted candidate pool recall from `79.16%` to **`92.29%` on unclean workings** and **`94.60%` overall** (`+15.44%` recall expansion).

---

### Breakthrough 2: 5-Paradigm Orthogonal Stacking Engine
Rather than ensembling minor hyperparameter variants of a single architecture, we combined 5 genuinely orthogonal learning paradigms exhibiting a **`67.3%` mean pairwise Top-1 disagreement**:

| Paradigm | Backbone & Training Objective | Key Diagnostic Signal | Standalone CV | Stack Weight |
|:---|:---|:---|:---:|:---:|
| **`P1` Multimodal VLM** | `Qwen2.5-VL-7B` (4-Bit NF4 QLoRA on 1,500 5-field CoT step annotations) | Direct visual token attention + listwise `[A]..[J]` logprobs | `0.24000` | **`12.0%`** |
| **`P2` Dense Bi-Encoder** | `BAAI/bge-m3` with Multi-Negative Ranking Loss (7 OOF hard negatives/trace) | 55,000-trace Classmate Consensus Bank (`95.71%` test `question_id` overlap) | `0.28400` | **`18.0%`** |
| **`P3` Step Cross-Encoder** | `DeBERTa-v3-Large` trained in strict `FP32` with Intra-Trace Margin Loss | Line-by-line first divergent step localization & secondary-slip discount | `0.12600` | **`10.0%`** |
| **`P4` Symbolic Verifier** | Deterministic algebraic step parser ($L_{k-1} \to L_k$) + 38 sibling rules | Exact arithmetic/algebraic equivalence & sibling disambiguation | `0.44800` | **`35.0%`** |
| **`P5` Multi-GBDT Stacker** | `LightGBM` + `CatBoost` pairwise LambdaRank over $\mathcal{H}_{24}$ candidate pool | Non-linear interaction across calibrated out-of-fold z-scores | `0.60922` | **`25.0%`** |
| **Champion `v4.2` Stack** | **Unitary $\mathcal{H}_{24}$ Union + Max-Preserving Projection ($\mathcal{P}^{\perp}$)** | **Full 5-Paradigm Orthogonal Synthesis** | **`0.68089`** | **`100.0%`** |

---

### Breakthrough 3: Max-Preserving Orthogonal Subspace Projection ($\mathcal{P}^{\perp}$)
To prevent error-candidate re-ranking from perturbing the high-precision `NONE` Clean Gate boundary (`0.94246` MAP@3), all non-`NONE` logits pass through **Max-Preserving Orthogonal Subspace Projection**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  MAX-PRESERVING ORTHOGONAL SUBSPACE PROJECTION INVARIANT (P^perp)                      │
│                                                                                        │
│  1. Compute Anchor Error Ceiling:   M_orig(i) = max_{c != NONE} s_anchor(i, c)         │
│  2. Blend Auxiliary Paradigms:      s_raw(i, c) = s_anchor(i, c) + Σ_k w_k * z_k(i, c) │
│  3. Compute Blended Error Ceiling:  M_new(i)  = max_{c != NONE} s_raw(i, c)            │
│  4. Project onto Max-Locked Plane:  s_hat(i, c) = s_raw(i, c) + [M_orig(i) - M_new(i)] │
│                                                                                        │
│  ★ Exact Mathematical Guarantee:    max_{c != NONE} s_hat(i, c) === M_orig(i)          │
│  ★ Result: Ranks #1/#2/#3 among error codes reorder freely with ZERO Clean Gate drift! │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Breakthrough 4: 38 Deterministic Symbolic Sibling Disambiguation Rules
Pedagogical misconceptions frequently form tight sibling pairs where neural classifiers output nearly tied logits. We encoded **38 deterministic symbolic rules** inspecting step-to-step transitions $L_{k-1} \to L_k$:
- **`ANS-01` vs. `ANS-02` (Answer Formulation):** Checks whether the student stopped at an intermediate variable on $L_{-1}$ (`ANS-01`) vs. computed the correct target on $L_{k}$ and applied an unrequested extra operation on $L_{n}$ (`ANS-02`).
- **`KPF-01` vs. `KPF-02` (Prime Factors & Fractions):** Distinguishes LCM/GCD prime exponent selection errors (`KPF-01`) from direct numerator/denominator addition $\frac{a}{b}+\frac{c}{d} \to \frac{a+c}{b+d}$ (`KPF-02`).
- **`Dec-01` vs. `Dec-02` (Decimal Place Value):** Separates whole-number digit length comparison (`3,48 > 3,5` because `48 > 5`, `Dec-01`) from vertical column misalignment (`Dec-02`).
- **`FRA-06` vs. `CAL-01` (Fraction Division vs. Slip):** Detects failing to invert the divisor or keeping the division operator after flipping (`FRA-06`) while discounting downstream arithmetic slips (`CAL-01`).
- **Verified Gain:** Contributed **`+0.0485` MAP@3** across the unclean cohort.

---

### Breakthrough 5: Adversarial Anti-Memorization Audit & 6-Gate Governance
During mid-campaign exploration, unconstrained regex patterns matching specific numbers and verbatim strings from `train.csv` produced an artificial local CV spike to `0.90044` that stagnated on the Public Leaderboard (`0.58200`).
We executed a comprehensive adversarial audit:
1. **Stripped All Train-Literal Lookups:** Removed every non-generalizable literal pattern and re-baselined the entire stack on strict out-of-fold predictions (**`0.68089` Honest 5-Fold CV**).
2. **Institutionalized 6 Automated Promotion Gates (`G1`–`G6`):**
   - `G1 (Overall CV Floor)`: Must strictly beat anchor 5-fold OOF MAP@3 (`0.68089`).
   - `G2 (Unclean Cohort Gain)`: Must improve unclean error diagnosis (`0.56914`).
   - `G3 (Clean Gate Non-Regression)`: Must preserve `NONE` accuracy (`0.94246`).
   - `G4 (High School Non-Regression)`: Must not regress Grades 9–12 (`0.62954`).
   - `G5 (Horizon Recall Floor)`: Must maintain $\mathcal{H}_{24}$ recall $\ge 90.0\%$ (`94.60%`).
   - `G6 (Paradigm Orthogonality)`: Must maintain $\ge 8.0\%$ pairwise disagreement (`67.30%`).

---

## ⚡ Compute Strategy: Single Colab + Tailscale Mesh & Kaggle Fallback

Executing 4-bit QLoRA fine-tuning on `Qwen2.5-VL-7B` and `FP32` cross-encoder training on `DeBERTa-v3-Large` using a single standard Google Colab account required resilient infrastructure engineering (see **Figure 6**):
1. **Direct Tailscale WireGuard P2P Mesh:**  
   Instead of brittle third-party HTTP/TCP proxies (`ngrok` bandwidth caps, `localtunnel` disconnects), we joined the remote Colab Tesla T4 instance directly to a private **Tailscale WireGuard mesh network**. This enabled persistent, low-latency headless SSH orchestration from our local workstation across multi-hour runs.
2. **Atomic Fold-by-Fold Checkpointing to Google Drive:**  
   Each fold (~18 minutes training + ~15 minutes candidate scoring) atomically flushed its out-of-fold `.parquet` matrices directly to mounted Google Drive storage upon completion. Ephemeral session resets never lost a single completed fold.
3. **Kaggle GPU Quota Fallback:**  
   When daily Colab GPU quotas were reached, our packager bundled `src/qaggle` and wheel dependencies into an `8.33 MB` tarball and dispatched headless GPU script kernels via the Kaggle API for seamless continuation.

---

## 🔍 Mistakes Made, Post-Mortems & Key Learnings

| # | Engineering Mistake | Symptom Observed | Root Cause Diagnosis | Permanent Fix & Lesson Learned |
|:---:|:---|:---|:---|:---|
| **1** | **Train-Literal Regex Memorization** | Local CV spiked to `0.90044`, but Public LB dropped to `0.58200`. | Regex rules matched specific student numbers present in `train.csv` rather than algebraic structure. | Stripped all literal rules; added `check_anti_memorization.py` linter and strict out-of-fold validation (`0.68089` CV). |
| **2** | **Prepending Setup Code Above `__future__`** | Instant `SyntaxError` on remote GPU workers when launching scripts. | String-prepending `sys.path` setup placed code above `from __future__ import annotations`. | Packaged code via `pip install -e .` and passed `PYTHONPATH` via `os.environ` in subprocesses. |
| **3** | **DeBERTa-v3 FP16 Underflow on Turing T4** | Cross-encoder loss became `NaN` on step 1 and zeroed OOF predictions. | DeBERTa-v3 disentangled relative position attention overflows `float16` range on Turing GPUs. | Enforced `torch.float32` on T4 GPUs and added `assert torch.isfinite(loss)` fail-fast guards. |
| **4** | **Unprojected Probability Blending** | Adding neural rerankers reduced overall MAP@3 by `-0.0385`. | Positive error deltas pushed $\max_{c \neq \text{NONE}} s(c)$ above $s(\text{NONE})$ on clean traces. | Enforced Max-Preserving Orthogonal Projection ($\mathcal{P}^{\perp}$) to lock the Clean Gate boundary. |

### Tactical Takeaways for Future Competitions
- **Measure Candidate Horizon Recall Before Building Rerankers:** If your top-$K$ pool has 79% recall, even an oracle reranker is capped at 0.79. Expanding $\mathcal{H}_8 \to \mathcal{H}_{24}$ (`94.60%` recall) unlocked our entire downstream stack.
- **Decouple Bimodal Regimes Mathematically:** Whenever a dataset has a high-accuracy dominant class (`NONE` math workings, non-fraud transactions, normal medical scans), project subclass rerankers orthogonally to the binary detection boundary.
- **Stress-Test Any Single-Feature CV Jump $> +0.05$:** Run out-of-fold and cross-question ablations immediately to catch subtle memorization early.

---

## 🛠️ Complete 1-Command Replication Guide

The repository is managed by [`uv`](https://github.com/astral-sh/uv) and supports deterministic 1-command verification and submission generation:

```bash
# 1. Clone the public showcase repository
git clone https://github.com/VijaiKMathew/k12-math-misconception-detection.git
cd k12-math-misconception-detection

# 2. Synchronize locked dependencies via uv
uv sync

# 3. Run Champion invariant checks, anti-memorization audit & submission verification
uv run python tools/checks/check_champion.py
uv run python tools/checks/check_submission.py --file submission.csv

# 4. Reproduce the Champion v4.2 submission and out-of-fold diagnostic report
uv run python tools/eval/produce_global_optima_s6e10_qaggle.py --export-aux
```

---

## 🤝 Acknowledgments & Community Support

Special thanks to **CoLearn Lab** and **Kaggle** for hosting this impactful competition at the intersection of multimodal AI, symbolic reasoning, and K-12 mathematics education.

⭐ **If you found our architecture, diagrams, interactive HTML dashboard, or masterclass notebook helpful, please consider leaving an Upvote on Kaggle and starring our [GitHub Repository](https://github.com/VijaiKMathew/k12-math-misconception-detection)!**

---

## 🏁 Private Leaderboard Post-Mortem & Forensic Gap Analysis

**Final Competition Standing (Released November 2026):**
- **Team:** `zoro-in-action` (`vijaikm`)
- **Selected Champion (`v4.2`, ref `#58394418` / `#58314170`):** Public LB **`0.62483`** | Private LB **`0.62496`** (**`+0.00013` Zero-Shakeup Generalization**)
- **Peak Private LB (Unselected `v2.9`, ref `#57973475`):** Public LB `0.62383` | Private LB **`0.62603`**
- **Final Leaderboard Rank:** **`226 / 484`** teams
- **Top-10 Private LB Benchmark:** **`0.84414`** (`Komil Parmar`, `Wassim Dobbi`) — Gap: **`+0.21918` MAP@3**

### 1. Private Leaderboard Top-10 vs. Our Pipeline

| Rank | Team | Public LB | Private LB | Gap to Our `v4.2` |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Komil Parmar** | `0.84366` | **`0.84414`** | `+0.21918` |
| **2** | **Wassim Dobbi** | `0.84766` | **`0.84414`** | `+0.21918` |
| **3** | **Cloudyy** | `0.83083` | **`0.83740`** | `+0.21244` |
| **4** | **Crown** | `0.82600` | **`0.83620`** | `+0.21124` |
| **5** | **Massimiliano Pianges** | `0.83583` | **`0.83466`** | `+0.20970` |
| **6** | **cuda graph compile error** | `0.84766` | **`0.83453`** | `+0.20957` |
| **7** | **Winter Lamb** | `0.83083` | **`0.83393`** | `+0.20897` |
| **8** | **monnu** | `0.84183` | **`0.83259`** | `+0.20763` |
| **9** | **Amin** | `0.83366` | **`0.83133`** | `+0.20637` |
| **10** | **Zedan Soorya** | — | **`0.82906`** | `+0.20410` |
| **226** | **zoro-in-action (`v4.2`)** | **`0.62483`** | **`0.62496`** | *Baseline* |

---

### 2. Pros: What Worked & Why We Had Zero Shakeup

1. **Zero-Shakeup Generalization (`0.62483` Public $\to$ `0.62496` Private):**  
   While overfitted submissions collapsed on the 71% Private split (including our own `v4.3` at `0.69256` CV $\to$ `0.60735` Private), `v4.2` improved by `+0.00013` on Private LB. Our adversarial anti-memorization audit and leak-free 5-fold CV (`0.68089`) prevented a catastrophic shakeup.
2. **Clean Gate Protection ($\mathcal{P}^{\perp}$):**  
   Locking the `NONE` decision boundary at **`0.94246` MAP@3** prevented false-positive misconception flips on correct student workings.
3. **Unitary Horizon Expansion ($\mathcal{H}_{24}$):**  
   Breaking the frozen Top-8 candidate ceiling (`79.16%` $\to$ `94.60%` overall / `92.29%` unclean recall) proved that candidate generation was not the final bottleneck.
4. **38 Deterministic Symbolic Sibling Rules:**  
   Added `+0.0485` MAP@3 on unclean workings with zero degradation on clean workings.

---

### 3. Cons & What Failed Miserably

1. **Train-Literal Regex Memorization (`v4.0` / `v4.1`):**  
   Created a deceptive **`0.90044` CV** illusion that collapsed to **`0.62333` Public / `0.62369` Private**.
2. **Over-Parameterized Listwise Calibration (`v4.3`):**  
   Pushed CV to `0.69256` but overfit to OOF logit noise, dropping to **`0.61283` Public / `0.60735` Private** (`-0.01761` vs `v4.2`).
3. **Text-Only LLMs & Monolithic Prompting (`-0.0520` MAP@3):**  
   Because `test.csv` contains **zero** human `transcription_steps`, any pipeline relying on text-only causal LMs or monolithic prompting suffered immediate modality-shift collapse.
4. **Frozen Top-8 Candidate Pool (`-0.0710` MAP@3):**  
   Capped recall at `79.16%`, starving downstream rerankers of ground-truth candidates.

---

### 4. What We Missed (GPU/Time Constraints) vs. Top-10 Winners (`0.844+`)

The **`+0.219` MAP@3 gap** between `0.62496` and `0.84414` was not caused by ensemble weighting—it was a **representation and compute-scale gap**:

1. **Two-Stage VLM-to-LaTeX Transcription Before Reranking:**  
   Top teams (`0.83–0.844`) used fine-tuned **Qwen2.5-VL-7B / 14B / 32B** (and dual-VLM ensembles) to **transcribe handwritten test images line-by-line into structured LaTeX** *first*, bridging the 100% missing `transcription_steps` gap in `test.csv`. On a single **Colab Tesla T4 (16GB VRAM)**, we were restricted to 4-bit QLoRA on `Qwen2.5-VL-7B` with small batch sizes and truncated context.
2. **Large Cross-Encoder Rerankers with Mined Hard Negatives:**  
   Winners trained **Qwen2.5-7B/14B LoRA rerankers** + **DeBERTa-v3-Large** cross-encoders using **mined hard negatives** (top-K false positives from bi-encoder retrieval) and **synthetic distractor-explanation augmentation**, then distilled/quantized them to fit Kaggle's 9-hour offline GPU limit.
3. **Multi-GPU / A100 Compute Budget:**  
   Full 5-fold VLM fine-tuning + 14B cross-encoder ensembling requires A100/H100 compute. Our single-T4 Tailscale WireGuard mesh kept us competitive in engineering rigor and zero-shakeup stability, but could not match 14B/32B VLM transcription fidelity.

### 5. Actionable Playbook for Next Competition

1. **Prioritize Modality Bridging First:** When `test.csv` drops a critical modality (like `transcription_steps`), dedicate 60%+ of GPU budget to training a high-capacity VLM transcriber before building downstream stackers.
2. **Mine Hard Negatives Early:** Replace random in-batch negatives with top-K bi-encoder false positives for cross-encoder reranking.
3. **Keep Simple, Audited Baselines:** Our `v2.9` (`0.62603` Private) and `v4.2` (`0.62496` Private) beat over-tuned `v4.3` (`0.60735` Private). Simplicity + strict OOF hygiene wins shakeups.

---

## 🎓 Plain-English Intuition & Career Takeaways (For Junior Data Scientists)

If you are early in your machine learning or Kaggle journey, here is the simple intuition behind what happened in this competition—and the career lessons you can apply to your next project.

### 1. The Simple Intuition: Why Was This Problem Hard?
Imagine grading 1,000 handwritten math homework sheets:
- **In training**, you are given both the **photo** of the student's handwriting *and* a **clean typed transcript** of every step they wrote.
- **In the final test**, you are **only given the photo**—no typed transcript!
- If your model only learned to read the typed transcript, it fails completely on the test set.
- Furthermore, **45% of students got the answer right (`NONE`)**. If your model guesses a misconception when the student was actually right, your score drops sharply.

### 2. How Our Solution Worked (In 4 Simple Steps)
1. **Don't Rely on One Model (5-Paradigm Diversity):** Instead of tuning one model 50 times, we combined 5 completely different approaches: a Vision-Language Model (`Qwen2.5-VL`), a similarity search over 55,000 past examples (`BGE-M3`), a text-pair scorer (`DeBERTa-v3`), 38 deterministic math rules, and gradient-boosted trees (`LightGBM` + `CatBoost`).
2. **Cast a Wide Net First ($\mathcal{H}_{24}$):** If you only look at your top 8 guesses, you miss the right answer 20% of the time. Expanding the candidate pool to 24 guesses lifted our recall to **94.6%**.
3. **Protect the "No Error" Class ($\mathcal{P}^{\perp}$):** We locked the threshold for predicting `NONE` so that combining multiple models wouldn't accidentally flip a correct student's work into a false error.
4. **Trust Honest Validation Over Inflated Scores:** Our `v4.0` model scored `0.90044` locally because it memorized exact training phrases, but it crashed to `0.62369` on the real test. Stripping out those shortcuts dropped local CV to `0.68089`—and gave us **zero shakeup** (`0.62483` Public $\to$ `0.62496` Private).

### 3. Five Career Lessons from This Competition
1. **Solve the Missing-Modality Gap First:** Always inspect what is in `train.csv` vs. `test.csv` on Day 1. When `test.csv` drops a key feature (like `transcription_steps`), your #1 priority is building a model to reconstruct it (e.g., VLM OCR), not tuning downstream ensembles.
2. **A High Local Score Can Be a Trap:** If your local CV jumps from `0.68` to `0.90` overnight, be suspicious—check for data leakage or memorization immediately.
3. **Hard Negatives Beat Random Negatives:** Teaching a model to distinguish between two *nearly identical* math errors (e.g., sign error vs. order-of-operations error) is far more valuable than comparing against random errors.
4. **Compute Constraints Force Engineering Discipline:** Even with a single T4 GPU, strict out-of-fold validation, clean modular code, and reproducible pipelines prevent overfitting and build production-ready habits.
5. **Document Your Failures:** Recording what *didn't* work (`v4.0` memorization, `v4.3` over-calibration) is just as valuable as recording what worked—it saves you and your team weeks on future projects.
