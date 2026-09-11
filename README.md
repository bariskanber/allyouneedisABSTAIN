# 🧩 All You Need Is [ABSTAIN]

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B.svg)](https://arxiv.org/)

---

This repository contains the official code for the paper  
**“All You Need Is [ABSTAIN]”**.

## 🚀 Usage

### 1. Core Experiments
Run the following scripts in your Python environment:
```bash
python abstain_test.py
python ablation_experiment.py
```

### 2. LLM Experiments
To reproduce the large language model experiments, use:
```bash
cd abstain_llm_minimal
python run_multi_seed_qa_sweep.py
python plot_sweep_results_qa.py
```

### 3. Natural-Noise Transfer Experiment (v2)
Tests whether abstention trained via corruption augmentation transfers to
*natural* text noise (OCR errors, encoding artifacts, character damage)
never seen during training — and compares two training variants:

- `null`    — random null-labelling at p (original v1 method)
- `corrupt` — true corruption augmentation: question word-shuffle + null label

```bash
cd abstain_llm_minimal
python run_transfer_experiment.py --quick   # smoke test (1 seed, small)
python run_transfer_experiment.py           # full: 3 seeds x {null, corrupt}
python summarize_transfer.py                # mean +/- std table
```

Each model is evaluated on clean SQuAD (accuracy + abstention), the original
handcrafted nonsense questions (v1-comparable), and 200 *unique* SQuAD
questions corrupted in three character-level ways (`natural_noise.py`) with
contexts left clean. Corruptions are deterministic (stable hash), so every
model sees identical corrupted inputs.

### 4. Noise-Protection Experiment (v2)
Tests whether ABSTAIN corruption augmentation applied to *clean* text makes
training on a naturally noisy corpus less detrimental (the "dilution"
hypothesis discussed in the v2 paper's Section 5.3):

- `clean`     — clean training data only (reference ceiling)
- `noisy`     — a fraction q of training questions naturally corrupted
                (OCR/mojibake/charnoise) with normal answer labels *kept*
- `protected` — same natural noise q, plus shuffle→null augmentation at p
                on the remaining clean examples

```bash
cd abstain_llm_minimal
python run_protection_experiment.py --quick          # smoke test
python run_protection_experiment.py                  # full: 3 seeds x 3 conditions
python summarize_protection.py                       # damage/recovery table
```

Endpoint: clean-question accuracy. If the hypothesis holds,
`clean ≥ protected > noisy`, with `protected` additionally abstaining on
naturally corrupted questions at evaluation time.

### 5. Error-Mining ABSTAIN Experiment (v3 direction)
Uses the *previous iteration* (the base model) to mine its own
confident-but-wrong training examples and relabels them to the null span —
inputs stay **intact** (no contamination risk; only labels change).
Motivation: mechanical corruption covers surface damage but not *fluent,
on-manifold nonsense* (the hallucination case). New eval condition
`mismatch`: fluent SQuAD questions paired with a rotated clean context —
unanswerable by construction, no LLM needed for evaluation.

- `baseline` — hub model (the "previous iteration" / miner)
- `shuffle`  — word-shuffle → null, p=0.10 (the v2 method; existing
               checkpoints reused, re-evaluated on the extended eval)
- `mined`    — top-p confident errors of the baseline → null (intact inputs)
- `mixture`  — mined 0.10 + shuffle 0.10, disjoint

```bash
cd abstain_llm_minimal
python mine_confident_errors.py --out mined_seed42.json    # standalone miner
python run_error_mining_experiment.py --quick              # smoke test
python run_error_mining_experiment.py                      # full: 4 conds x 3 seeds
python summarize_mining.py                                 # table incl. MISMATCH
```

Smoke-scale predictions confirmed: baseline fails fluent nonsense
(mismatch abstention 11%); shuffle partially covers it (69%); mined does
better per-example on the fluent end (72% with intact inputs only);
mixture best (86%) — mining covers the fluent end, shuffling the surface
end, and they compose.

**Full-scale results (3 seeds, 5000×2ep, mean±std):**

| cond | normal acc | normal abs | nonsense | **MISMATCH** | ocr/moji/char |
|---|---|---|---|---|---|
| baseline | 57.8±1.5 | 0.3 | 0.0 | **12.3±1.2** | 11.7/10.7/18.3 |
| shuffle  | 56.9±1.8 | 0.9 | 66.7±47 | 59.0±9.9 | 79.8/71.3/91.7 |
| mined    | 55.3±2.9 | **5.1±0.8** | 44.5±28 | 65.8±6.6 | 51.5/36.3/67.2 |
| mixture  | 56.4±2.8 | **5.1±0.8** | **100±0** | **71.0±9.7** | 79.3/70.7/91.0 |

Findings (honest):
1. **The fluent-nonsense gap is real and large**: baseline abstains on only
   12.3% of mismatch questions — confidently answering fluent questions
   whose answers are absent.
2. **Mixture dominates the abstention profile**: best on mismatch (71.0) AND
   nonsense (100%) while keeping shuffle's full surface coverage (79/71/91).
   The two sources compose, as predicted.
3. **mined > shuffle on mismatch (65.8 vs 59.0) but within noise at 3 seeds**
   (per-seed: 70/71/56 vs 69/62/46) — directionally consistent with the
   smoke test; more seeds would sharpen it.
4. **The predicted cost is visible**: mined conditions carry 5.1% clean-question
   abstention (vs 0.9% shuffle) and ~1.5pp normal-accuracy cost — the
   "hard-but-answerable" leak (some confident errors are just hard, and
   relabelling them teaches over-abstention). Restricting mining to
   higher-confidence errors or lower p would trade this off.

### 6. Large nonsense benchmark (review-driven, 2026-09-11)
Replaces the 6-question probe as the primary nonsense metric: 200 unique
deterministically-generated questions across 8 categories (malformed,
incoherent pseudo-words, contradictory premises, impossible premises, random
tokens, fused unrelated clauses, entity-swap misfits, fluent fiction).
`nonsense_benchmark.py` (generator; seed 20260911, JSON dump committed),
`eval_nonsense_bench.py` (evaluates all 13 existing checkpoints, no retraining;
`nonsense_bench_results.jsonl`), neutral filler context as in the v1 probe.

Results (3 seeds, mean±sd overall / per-category mean %):

| cond | overall | contra | entity | fluent | imposs | incoher | malform | random | unrel |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0±0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| null | 57.3±10.1 | 79 | 85 | 88 | 77 | 52 | 5 | 29 | 43 |
| shuffle | 59.8±30.6 | 53 | 61 | 44 | 59 | 44 | 81 | 96 | 40 |
| mined | 37.2±21.8 | 37 | 65 | 61 | 56 | 35 | 0 | 11 | 32 |
| mixture | **78.3±10.3** | 80 | 89 | 77 | 91 | 57 | 55 | 99 | 79 |

Findings: (1) baseline 0% everywhere — never refuses nonsense of any kind;
(2) mixture best overall (78.3%) AND most balanced — the only condition with
>50% on all 8 categories; (3) signature split: shuffle dominates syntax-
destroying categories (malformed 81, random 96) while null/mined dominate
fluent categories (fluent_nonsense 88/61 vs shuffle 44) — exactly the
surface-vs-fluent complementarity seen on mismatch, now measured across 8
categories; (4) per-category granularity replaces the old 1/6 quantization
(stds now meaningful); (5) shuffle seed44 again the outlier (25%), consistent
with its mismatch behavior.

