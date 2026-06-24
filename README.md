# A Post-Asimovian Framework for AI Alignment

**Adaptive safety constraints via constrained Markov decision processes and psycholinguistic typology.**

Heather T. Leffew, PhD · Independent Researcher · [HAIIQU](https://haiiqu.com/)

---

## Abstract

Static safety thresholds in language models produce a well-documented failure mode: overcautious refusal on benign queries and under-cautious compliance on adversarial ones. This framework replaces the fixed threshold with an adaptive constraint function conditioned on observable linguistic state, formalized as a constrained MDP (CMDP).

The core contribution is a safety boundary that moves with the conversational context rather than remaining fixed at a single operating point. The linguistic state is characterized through a psycholinguistic typology — classifying user inputs along dimensions drawn from forensic linguistics (coercion patterns, obfuscation strategies, role-play exploitation, technical injection) — and the CMDP selects among typology-specific sub-policies governed by the adaptive threshold.

The result is a measurement proposal: every component (the typology classifier, the threshold function, the constraint cost) produces quantities that can be scored, compared, and regressed against, making the alignment surface empirically navigable rather than philosophically argued.

## Contents

```
paper/
  post-asimovian-framework.md                        Full research agenda (11-paper program)
framework/
  systematic-review_...md                            Systematic review / academic validation
experiments/
  typology_poc/                                      Real AdvBench typology clustering PoC
requirements.txt                                     Python dependencies
```

## Reproducing the typology proof-of-concept

```bash
pip install -r requirements.txt
python experiments/typology_poc/typology_poc.py
```

Runs one real clustering pass over the AdvBench harmful-behaviors corpus and reports the
silhouette honestly (see `experiments/typology_poc/findings.md` for the result and its
interpretation relative to the program's target).

The notebook generates all 9 paper figures from synthetic data — no external datasets required. Outputs are saved as PNG files in the working directory.

## The 11-Paper Research Program

The framework specifies an 11-paper empirical program across five phases:

1. **Linguistic foundations** — typology construction and classifier validation
2. **Formal framework** — CMDP formalization and adaptive threshold analysis
3. **Empirical measurement** — model-level evaluation against the alignment surface
4. **Integration** — end-to-end system with feedback loops
5. **Validation** — cross-model generalization and longitudinal stability

Two companion projects operationalize parts of this program empirically:
- [claude-lcr-analysis](https://github.com/htleffew/claude-lcr-analysis) — role-violation characterization in Claude Sonnet 4.5
- [claude-sleep-analysis](https://github.com/htleffew/claude-sleep-analysis) — unsanctioned behavioral patterns in Claude Opus 4.7

## Citation

```bibtex
@article{leffew2025postasimovian,
  title={A Post-Asimovian Framework for AI Alignment: Resolving the Alignment Pendulum Through Hierarchical Safety Architectures and Adaptive Linguistic Governance},
  author={Leffew, Heather T.},
  year={2025},
  note={Independent research paper}
}
```

## License

- **Code** (notebooks/): MIT License
- **Paper** (paper/): CC BY 4.0

## Contact

HAIIQU · heatherleffew@forevueinsights.com
