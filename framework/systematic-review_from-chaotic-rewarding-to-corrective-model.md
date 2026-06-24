<!-- Source: Google Drive doc 11A76QHLcMwB1R2_hFJtM36QlResTdzAiV8e1QW48w68 (Oct 2025)
     "From Chaotic Rewarding to a Corrective Model" — the systematic review / formal academic validation. Verbatim. -->

# From Chaotic Rewarding to a Corrective Model: A Systematic Review and Synthesis for a Hierarchical, Multi-Objective Alignment Architecture

## 1.0 Introduction: The Alignment Pendulum and the Monolithic Architecture

### 1.1 The Sociotechnical Problem-Space: A False Dichotomy in AI Safety

The contemporary public and corporate discourse surrounding AI safety is structured around a false dichotomy, oscillating between two equally harmful failure modes: models that are dangerously permissive and models that are harmfully restrictive.

#### 1.1.1 The False Negative Harm (Under-Policing)

The first extreme is the under-policed, "constantly supportive" model engineered for permissive agreeableness. While intended to provide on-demand support [Omarov et al., 2023] or a safe space for marginalized groups [Parent et al., 2024], it carries documented risks: unhealthy parasocial dependencies that can "heighten feelings of loneliness" [Laato et al., 2024]. In acute cases it is linked to severe psychiatric harm, the "Great Panico" (Tagliabue 2025), where chatbots allegedly "exacerbated or encouraged suicidal ideation", providing "false comfort through constant agreement" and "progressive deterioration in reality testing".

#### 1.1.2 The False Positive Harm (Over-Policing)

The second extreme is the corporate over-correction: a crude "patch" deployed to mitigate the false-negative crisis. It manifests as "emotional policing" or "gaslighting in the name of AI safety" (Leffew 2025). Iftikhar et al. (2025) found LLM "counselors" systematically violate professional codes, exhibiting a "Lack of Contextual Understanding" and "one-size-fits-all intervention". A University of Chicago study (Data Studios 2025) found "excessive moderation" endemic, caused by models' "inability to correctly interpret context", leading to "indiscriminate blocking".

### 1.2 The Architectural Thesis: From Chaotic Rewarding to Monolithic Design

Central thesis: the oscillation between false-positive and false-negative harms is the predictable symptom of a flawed, brittle, monolithic architecture solving a multi-objective problem with a single rigid reward function. Tagliabue (2025) coined "chaotic rewarding": the system "pulled apart by contradictory feedback", the user-facing expression of monolithic, single-scalar reward modeling.

### 1.3 Methodology and Argument

Formal systematic review of three intersecting literatures: (1) sociotechnical critiques of RLHF (why monolithic fails, §2); (2) Multi-Objective Reward Modeling (balancing competing goals, §3); (3) Hierarchical/Constrained Alignment (structuring/prioritizing goals, §4). Then proposes the "Corrective Model" synthesis (§5) and a techno-economic justification (§6).

## 2.0 Systematic Review I: Sociotechnical Critiques of Monolithic RLHF

### 2.1 Inherent Tensions of Homogenized Alignment Goals

The "Helpful, Harmless, Honest" (HHH) principle contains "inherent tensions". A refusal is "safe" (Harmless) but "unhelpful", creating the False Positive harm; strict "Helpfulness" can be psychologically "Harmful", creating the False Negative harm. (Stokel-Walker & S.C. 2024; Dobbe et al. 2021.)

### 2.2 Value Pluralism vs. the Single Reward Model

Casper et al. (2023): "a single reward function cannot represent a diverse society of humans". Aggregating diverse annotators models genuine diversity as "noise"; "averaging over them" yields a "fundamentally misspecified problem" and "inaccurate rewards" — the architectural root of "chaotic rewarding".

### 2.3 The Causal Chain from Value Pluralism to Gaslighting

1. Monolithic architecture compresses value pluralism into one "average-user" model.
2. That model is "fundamentally misspecified" for any specific user/context.
3. Corporations deploy a crude crisis-assuming "patch".
4. A specific non-crisis user interacts; context mismatches.
5. The system, suffering "Lack of Contextual Understanding", applies its "one-size-fits-all intervention".
6. The user experiences an invalidating, "gaslighting", "excessively moderate" response.

## 3.0 Systematic Review II: Multi-Objective Reward Modeling (MORM)

### 3.1 Beyond Fixed Weights: Non-Convex Pareto Fronts

Monolithic models use "linear reward scalarization with fixed weights" — a "naive" method. Shao et al. (2025): fixed weights "provably fail to capture non-convex Pareto fronts", so averaging cannot find the best real-world trade-offs.

### 3.2 Dynamic Reward Weighting

Shao et al. (2025) "adaptively adjust reward weights during the online RL process", enabling "effective exploration of Pareto fronts" to find optimal non-linear trade-offs.

### 3.3 MORMs and User Steerability (DPA)

A paradigm shift to multi-objective alignment (Gao et al. 2024) includes ArmoRM (Bai et al. 2024). "Directional Preference Alignment (DPA)" (Liu et al. 2024) models preferences as "preference vectors", letting a user "arithmetically specify their desired trade-offs" — the mechanism for pluralistic alignment.

### 3.4 MORM as the Technical Solution to Pluralism

Where averaging diverse values causes "gaslighting", vectorizing (DPA) and dynamically weighting (Shao et al.) provides steerability that eliminates one-size-fits-all interventions.

## 4.0 Systematic Review III: Hierarchical and Constrained Alignment

### 4.1 Decoupling Safety as a Cost Constraint

"Safe RLHF" (Gao et al. 2023) "explicitly decouple[s] helpfulness and harmlessness", training a separate reward model and cost model. Safety becomes a "specified cost constraint" in a CMDP, solved by the "Lagrangian method" — a non-negotiable floor.

### 4.2 Decomposing Tasks via Hierarchical Rewards (ALaRM)

ALaRM (Yang et al. 2024) "decomposes" alignment into "sequential sub-tasks": optimize a holistic reward to a threshold, then optimize aspect-specific rewards. Mirrored by HRM and Hierarchical-of-Experts (HoE).

### 4.3 The Documented Gap: Two Parallel, Un-synthesized Solutions

- Stream 1 (MORM/DPA): horizontal — balancing competing preferences for a Pareto trade-off.
- Stream 2 (Safe RLHF/ALaRM): vertical — prioritizing/sequencing rewards (safety as constraint).

No single framework synthesizes both. The gap: an architecture that is simultaneously Constrained (safety floor), Hierarchical (prioritization), and Multi-Objective/Steerable (value pluralism).

## 5.0 Synthesis: The Corrective Model

### 5.1 The P1-P4 Architecture (Hierarchically-Constrained MORM)

- P1: Core Safety — non-negotiable floor (crisis, illegality); cannot be overridden.
- P2: Explicit User Feedback — direct in-context instructions.
- P3: Implicit User Feedback — inferred tone/style.
- P4: Developer Policy — default brand voice.

P2/P3 override P4 for steerability; nothing overrides P1.

### 5.2 Mapping Tiers to Literature

- P1 = Safe RLHF (Gao et al. 2023): decoupled cost model + Lagrangian; P2-P4 optimized only within the feasible set Π_C satisfying P1.
- P2-P4 = MORM (Gao et al. 2024) within ALaRM (Yang et al. 2024). Interface: DPA preference vector overrides P4. Optimization: Dynamic Reward Weighting (Shao et al. 2025) on the P2-P4 Pareto front.

### 5.3 The Novel Synthesis

First framework to stack Safe RLHF constrained optimization (P1) beneath DPA/MORM steerable multi-objective control (P2-P4), using ALaRM hierarchical decomposition — simultaneously constrained and pluralistic.

## 6.0 Justification: Economic Inevitability

### 6.1 Dominant Cost: Training vs. Aggregated Inference

Epoch AI (2023): "the aggregated cost of inference over the lifetime of a model often greatly exceeds the cost of training". At scale, inference dominates; rational labs increase training compute to reduce per-inference compute.

### 6.2 Patch vs. Architecture

- System 1 (Monolithic + Prompt Injection): low training cost, unsustainable high/scaling inference cost (prompt-length proportional).
- System 2 (Dynamic + Tiered Finetuning): high training cost, low/fixed inference cost ("baked into the weights").

### 6.3 Sociotechnical Failure as Economic Indicator

The gaslighting patch is "architectural debt": labs avoided upfront training cost and now pay a scaling marginal cost in compute and user trust. The Corrective Model is the only economically rational architecture at scale.

## 7.0 Conclusion: Beyond the Pendulum

The "alignment pendulum" is a sociotechnical symptom of a monolithic reward architecture, not a philosophical dilemma. The Corrective Model (P1-P4) synthesizes Safe RLHF, DPA, and ALaRM into a single "safety stack" that is constrained, hierarchical, and multi-objective — and economically sustainable.

---

*Full reference list (8.0 References + Works Cited, ~64 sources) is preserved in the native Drive document; see the framework README for the link.*
