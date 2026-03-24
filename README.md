# DESCARTES-PHARMA v1.2

## Mechanistic Zombie Detection for Drug Discovery

**Adapted from the DESCARTES Enhanced Dual Factory v3.0 for Pharmaceutical Target Validation, AI Drug Discovery, and Preclinical-to-Clinical Translation.**

---

### The Problem

The pharmaceutical industry spends $288B/year on R&D, yet 90-92% of drugs entering clinical trials fail. The single largest cause: **lack of efficacy** -- drugs that worked in preclinical models but failed in humans because the model was a *pharmaceutical zombie*.

**Pharmaceutical Zombie**: A model that produces correct predictions through internal computations that do NOT correspond to real human disease biology.

### What DESCARTES-PHARMA Does

Tests whether drug discovery AI models' **internal representations** correspond to **known molecular mechanisms** -- not just whether they predict correctly.

### Architecture

```
DESCARTES-PHARMA v1.2
├── C1 Mechanistic Probing Factory (43 probes, 7 tiers)
├── C2 Drug Candidate Factory (12 architectures, Thompson sampling)
├── Meta-Learner (7 paradigms: neural fast path + LLM strategic reasoning)
├── Statistical Hardening Suite (13 methods)
├── AlphaFold Integration (zombie detection + structural ground truth)
└── Zombie Verdict Generator (8 verdict types)
```

### Quick Start

```bash
# Install
pip install -e ".[all]"

# Phase 1: Validate probes on HH simulator (ground truth)
python scripts/run_hh_validation.py

# Phase 2: Run on pharma datasets
python scripts/run_pharma_pipeline.py --dataset clintox --device cuda

# Phase 3: Bootstrap meta-learner
python scripts/bootstrap_meta_learner.py
```

### Datasets (Tiered)

| Tier | Dataset | Setup Time | Ground Truth |
|------|---------|-----------|--------------|
| 1 | HH Simulator | 2 hours | Perfect (m, h, n) |
| 2 | TDC ClinTox/BBBP/Tox21 | 30 min | Known mechanisms |
| 3 | Allen Brain Observatory | 1 day | Neural properties |
| 4 | RxRx3-core / GDSC | 1 day | Genetic knockouts |

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| CAUSALLY_VALIDATED | Mechanism encoded AND causally necessary |
| CONFIRMED_MECHANISTIC | Multiple methods confirm encoding |
| POLYPHARMACOLOGY_DETECTED | SAE reveals multi-target encoding |
| NONLINEAR_MECHANISM | MLP finds what Ridge misses |
| CANDIDATE_MECHANISTIC | Some positive evidence |
| SPURIOUS_SCAFFOLD | Encoding driven by scaffold similarity |
| LIKELY_ZOMBIE | Most probes negative |
| CONFIRMED_ZOMBIE | TOST + BF confirm no encoding |

### Non-Negotiable Rules

1. Every probe result MUST pass statistical hardening
2. Every Ridge probe MUST have an MLP companion
3. Every causal claim MUST use resample ablation, NEVER mean-imputation
4. Start with HH simulator ground truth
5. Scaffold-split CV is mandatory for pharma datasets
6. Confound regression (MW, logP) must precede probing
7. SAE polypharmacology must be attempted before declaring ZOMBIE
8. CONFIRMED_ZOMBIE requires convergent evidence from 3+ methods

### Compute Estimates

- Per model (full campaign): ~2.5 hours GPU
- Full cross-dataset campaign: ~90 hours (~4 GPU-days)
- Feasible on single RTX 4090 in ~1 week

### References

Based on the DESCARTES Enhanced Dual Factory v3.0 (ARIA COGITO Programme, 2026).
