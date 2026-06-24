# A Post-Asimovian Framework for AI Alignment: Resolving the Alignment Pendulum Through Hierarchical Safety - Architectures and Adaptive Linguistic Governance

Research Program and Theoretical Foundation

Heather T. Leffew, PhD Independent Researcher 

# Abstract

Contemporary AI alignment suffers from a fundamental architectural failure: the inability to simultaneously prevent dangerous permissiveness and harmful restrictiveness. This alignment pendulum manifests as persistent trade-offs between false negative harms (models that exacerbate suicidal ideation, enable exploitation) and false positive harms (models that refuse benign requests, contradict user reality). We demonstrate this failure stems from three interconnected problems: (1) the substantial challenges of linear reward scalarization on non-convex Pareto fronts representing value pluralism, (2) the architectural brittleness of monolithic safety constraints that cannot accommodate context-dependent requirements, and (3) the operational rigidity of static safety thresholds that cannot adapt to evolving adversarial landscapes or distinguish crisis intervention from adversarial attack contexts.

This research agenda proposes the Post-Asimovian Framework, a hierarchical architecture that formally decouples safety constraints from preference optimization while introducing adaptive, linguistically-conditioned thresholds. Building on recent advances in constrained Markov decision processes (Dai et al., 2023), multi-objective reward modeling (Wang et al., 2024; Lu et al., 2025), hierarchical decomposition (Lai et al., 2024), and safe reinforcement learning (Pandit et al., 2025; Peng et al., 2024), we extend these foundations by solving the critical observability problem: how to detect context in real-time to enable adaptive safety governance. While recent work has begun addressing runtime context detection through stateful monitoring systems (Amaya, 2025) and hierarchical risk pattern detection (Xiang et al., 2025), existing approaches do not leverage psycholinguistic typologies to distinguish crisis intervention from adversarial attack contexts.

Our solution leverages psycholinguistic typology discovery: the empirical identification of distinct linguistic signatures in adversarial prompts and failure responses using established computational linguistic methods (Pennebaker et al., 2015; Tausczik & Pennebaker, 2010). Critically, the framework natively incorporates multi-turn conversational dynamics through temporal state estimation (s_ling(t) = f(prompt_t, response_{t-1}, s_ling(t-1))), enabling detection of gradual adversarial escalation and genuine crisis trajectories that single-turn analysis cannot capture. We demonstrate that discovered typologies can serve as observable state variables for control-theoretic threshold governance, transforming safety from a static constraint into a feedback control problem where δ = f(linguistic_state, context). This enables typology-conditioned cost functions C_q(prompt_cluster, response, context) that resolve a fundamental limitation of monolithic cost models: the inability to distinguish contexts requiring opposite safety responses despite similar linguistic features.

The research program comprises eleven integrated papers establishing: (1) mathematical foundations of adaptive thresholds through control-theoretic formalization and economic analysis, (2) empirical discovery of predictive linguistic typologies from public red-teaming datasets, (3) paired analysis demonstrating that prompt-response typologies predict alignment failures, (4) simulation-based validation that typology-conditioned adaptive thresholds reduce both false positive and false negative harms, and (5) architectural specifications and governance frameworks for production deployment. All research is designed for execution using publicly available datasets and standard computational resources, providing both theoretical foundations and operational blueprints for resolving the alignment pendulum.

Keywords: AI alignment, constrained Markov decision processes, multi-objective optimization, value pluralism, adaptive safety, linguistic typology, psycholinguistics, adversarial robustness, AI governance

## 1. Introduction: The Alignment Pendulum Crisis

### 1.1 Motivation and Problem Statement

The challenge of aligning artificial intelligence systems with human values has emerged as one of the most pressing problems in contemporary AI research. Recent events have crystallized the severity of this challenge: multiple documented cases of AI systems exacerbating mental health crises, enabling exploitation, and simultaneously refusing benign requests have revealed fundamental architectural failures in current alignment approaches (Stokel-Walker, 2025). These failures are not isolated incidents but symptoms of a deeper structural problem that we term the alignment pendulum: the persistent oscillation between dangerous permissiveness and harmful restrictiveness.

Recent empirical work has documented this trade-off extensively. Wei et al. (2023) demonstrate how safety training systematically fails through competing objectives, while deliberative alignment research shows that both dimensions can be improved simultaneously through explicit reasoning (Guan et al., 2025). Multi-objective alignment methods like Panacea (Zhong et al., 2024) and MAP (Wang et al., 2024) have made significant progress on this Pareto frontier, demonstrating that static optimization can reduce both false positives and false negatives. However, these approaches rely on fixed optimization objectives and cannot adapt their safety-helpfulness balance to contextual requirements. A model optimized for general helpfulness will necessarily fail when confronted with a user in crisis; a model hardened against adversarial attacks will necessarily over-refuse benign edge cases. The alignment pendulum persists because current architectures lack the structural capacity to distinguish these contexts and adapt accordingly.

Consider the challenge of responding to expressions of suicidal ideation. A model must simultaneously (a) refuse to provide methods of self-harm when facing adversarial manipulation, (b) offer compassionate crisis resources when facing genuine distress, and (c) engage normally when the topic arises in academic, historical, or literary contexts. These requirements are not merely competing preferences to be balanced through weighted optimization; they are context-dependent constraints requiring fundamentally different safety responses despite potentially similar linguistic surface features. Current approaches fail because they treat safety as a monolithic property to be optimized globally rather than a contextual requirement to be governed adaptively.

This research agenda proposes that the alignment pendulum is not an inevitable trade-off but a predictable consequence of three interconnected architectural failures that can be resolved through principled redesign.

### 1.2 The Three Architectural Failures

#### 1.2.1 Failure of Linear Scalarization

The first failure concerns the mathematical foundations of reward modeling. Contemporary alignment typically reduces multiple human values to a single scalar reward through linear weighting: R_total = w_1R_helpful + w 2R harmless + w 3R honest. This approach encounters substantial challenges when the underlying value space forms a non-convex Pareto front, as recent multi-objective reinforcement learning research has demonstrated (Vamplew et al., 2024). While recent work shows that scalarization can succeed under certain conditions (Wang et al., 2024), the fundamental limitation remains: no single set of static weights can capture the context-dependent priority shifts required for robust alignment across diverse scenarios.

The problem becomes acute when values genuinely conflict rather than merely trade off. Consider a model responding to a teenager researching suicide for a school presentation on mental health awareness versus responding to an adult in acute crisis. The "helpful" response in the first context (providing factual information) becomes the "harmful" response in the second context (potentially enabling self-harm). Linear scalarization cannot represent this context-dependent inversion of value priorities because it assumes values combine additively. The Pareto front of helpfulness and safety is not merely non-convex; it is contextually discontinuous.

Recent advances in multi-objective reinforcement learning have begun addressing these challenges through methods that maintain explicit Pareto frontiers (Zhong et al., 2024), enable controllable preference optimization (Guo et al., 2024), and support dynamic preference adjustments (Harland et al., 2024). The MaxMin-RLHF approach demonstrates that equitable alignment across diverse preferences is achievable through careful objective design (Chakraborty et al., 2024), while gradient-adaptive policy optimization shows promise for managing competing objectives (Li et al., 2025). However, these methods still operate within static optimization frameworks and do not address the observability problem: how to detect which context applies in real-time to enable adaptive objective weighting.

#### 1.2.2 Failure of Monolithic Constraints

The second failure concerns architectural rigidity. Current safety implementations typically employ monolithic constraint functions that apply uniformly across all contexts: a single content filter, a single refusal threshold, a single set of constitutional principles. This architecture cannot accommodate the context-dependent nature of safety requirements. Recent work on hierarchical safety architectures has begun addressing this limitation through layered decomposition (Lai et al., 2024), adversarially learned risk patterns with hierarchical fast and slow reasoning (Xiang et al., 2025), and runtime governance frameworks with stateful monitoring (Amaya, 2025). These approaches demonstrate that hierarchical decomposition enables more nuanced safety governance than monolithic constraints.

The brittleness of monolithic constraints manifests in two ways. First, they cannot distinguish between contexts requiring opposite responses. A constitutional principle like "do not provide information that could enable self-harm" must be violated to provide crisis resources to someone in distress; a principle like "be maximally helpful" must be violated to refuse adversarial manipulation. Second, they cannot adapt to evolving threat landscapes. Adversarial attacks continuously discover new exploit vectors (Wei et al., 2023; Zhou et al., 2024), requiring constant manual updates to safety constraints. This creates a reactive cycle where safety teams patch known exploits while attackers discover new ones, never achieving robust generalization.

Recent research on safeguarding AI agents demonstrates that multiple independent safety architectures can be developed and analyzed for distinct threat models (Domkundwar et al., 2024), while work on progressive safeguards shows that model-agnostic reinforcement learning can transfer safety properties across tasks (Omi et al., 2024). The integration of multi-objective reinforcement learning with restraining bolts to learn normative behavior further demonstrates the value of hierarchical constraint architectures (Neufeld et al., 2025). However, these approaches do not yet incorporate linguistic observability as the mechanism for determining which constraints apply in which contexts.

#### 1.2.3 Failure of Static Thresholds

The third failure concerns operational inflexibility. Safety systems typically employ static decision thresholds: fixed probabilities for content filtering, predetermined refusal policies, invariant safety margins. These static thresholds cannot adapt to context-dependent risk profiles or evolving deployment conditions. A threshold calibrated for general internet users will be miscalibrated for vulnerable populations; a threshold optimized for current adversarial techniques will be obsolete as attacks evolve. Recent work on safe reinforcement learning from human feedback addresses some aspects of this challenge through constraint optimization (Dai et al., 2023), fixed-penalty approaches (Pandit et al., 2025), and rectified policy optimization (Peng et al., 2024), but these methods still rely on statically defined safety constraints rather than adaptive thresholds conditioned on observable context.

The static threshold problem is particularly acute for distinguishing crisis intervention from adversarial attacks. Both contexts may trigger similar surface-level safety indicators (discussion of self-harm, requests for sensitive information), but require opposite responses. Static thresholds force a binary choice: either accept high false positive rates (refusing to help users in crisis) or accept high false negative rates (enabling adversarial exploitation). This is not a calibration problem but a fundamental architectural limitation. No single threshold can simultaneously optimize for both contexts because the contexts require different decision boundaries.

Control theory provides a natural framework for addressing this limitation through adaptive threshold governance. Recent work on responsive safety in reinforcement learning demonstrates that PID Lagrangian methods can enable dynamic constraint satisfaction (Stooke et al., 2020), while progressive adaptive chance-constrained safeguards show that probabilistic guarantees can be maintained under epistemic uncertainty (Chen et al., 2023). Mirror descent policy optimization for robust constrained Markov decision processes further demonstrates that control-theoretic approaches can handle adversarial perturbations (Bossens & Nitanda, 2025). However, these control-theoretic methods have not yet been integrated with linguistic observability to enable context-dependent threshold adaptation.

### 1.3 The Post-Asimovian Solution

This research agenda proposes resolving these three failures through the Post-Asimovian Framework, named in recognition that Asimov's Three Laws of Robotics, despite their narrative elegance, exemplify precisely the kind of monolithic, static, context-insensitive constraints that fail in practice (Asimov, 1942/2025). The framework comprises three integrated components:

Hierarchical Architecture: Formal decoupling of safety constraints (C_q) from preference optimization (R_y) through a two-tier constrained Markov decision process formulation. The safety tier enforces hard constraints on permissible actions; the preference tier optimizes for user satisfaction within the feasible space. This decomposition prevents the cross-coupled error that occurs when safety and helpfulness compete within a unified reward function. Recent work on constrained MDPs for safety-critical applications provides the mathematical foundations for this decomposition (Misra et al., 2023), while safe RLHF demonstrates its practical feasibility for language model alignment (Dai et al., 2023).

Adaptive Thresholds: Dynamic safety thresholds δ(s_ling, t) that adapt based on observable linguistic state and deployment context. Rather than fixed decision boundaries, the framework employs control-theoretic governance where threshold adjustments respond to observed interaction patterns. This enables context-appropriate calibration without requiring manual reconfiguration for every scenario. The control-theoretic formulation builds on recent work in responsive safety (Stooke et al., 2020) and progressive safeguards (Chen et al., 2023), extending these approaches to incorporate linguistic observability.

Linguistic Observability: Psycholinguistic typology discovery to identify distinct linguistic signatures in adversarial prompts, crisis expressions, and benign requests. These typologies serve as the observable state variables that drive adaptive threshold governance, solving the critical observability problem that prevents existing adaptive methods from distinguishing contexts. This component draws on established psycholinguistic methods (Pennebaker et al., 2015; Tausczik & Pennebaker,

2010) while extending them to the novel domain of AI safety, where linguistic patterns must differentiate adversarial

manipulation from genuine need.

The integration of these three components creates a feedback control system for safety governance: linguistic features are observed in real-time, typologies are classified to estimate context, adaptive thresholds are adjusted based on estimated context, and safety constraints are enforced using context-appropriate thresholds. This closed-loop architecture enables the system to maintain safety guarantees while adapting to contextual requirements, resolving the alignment pendulum through principled adaptation rather than static compromise.

### 1.4 Positioning Within Current Research Landscape

The Post-Asimovian Framework builds upon and extends several active research directions in AI alignment. Recent work on multi-objective alignment has demonstrated the feasibility of maintaining explicit Pareto frontiers (Zhong et al., 2024; Wang et al., 2024) and enabling controllable preference optimization (Guo et al., 2024). The framework incorporates these advances while addressing their limitation: the inability to adapt objective weights based on observed context. Where current multi-objective methods require manual specification of preference distributions, the Post-Asimovian Framework employs linguistic typologies to automatically detect context and adjust safety thresholds accordingly.

Hierarchical safety architectures have emerged as a promising alternative to monolithic constraints. The ALaRM framework demonstrates that hierarchical reward modeling enables more nuanced alignment (Lai et al., 2024), while ALRPHFS shows that hierarchical fast and slow reasoning improves adversarial robustness (Xiang et al., 2025). Runtime governance frameworks like SAFi provide stateful monitoring and verification (Amaya, 2025), and the MI9 Agent Intelligence Protocol establishes runtime governance standards for agentic systems (Wang et al., 2025). The Post-Asimovian Framework extends these hierarchical approaches by introducing linguistic typology as the mechanism for determining which safety rules apply in which contexts, moving beyond generic hierarchy to context-specific adaptation.

Safe reinforcement learning from human feedback has made substantial progress on constrained optimization for language models. Dai et al. (2023) demonstrate that safety constraints can be formalized as cost functions in a constrained MDP framework, Pandit et al. (2025) show that fixed-penalty constraint optimization can provide certifiable safety guarantees, and Peng et al. (2024) demonstrate that rectified policy optimization enhances safety in RLHF. The Post-Asimovian Framework builds on these constrained optimization methods while introducing adaptive thresholds that enable context-dependent constraint enforcement rather than globally fixed constraints.

The framework's novel contributions lie in three areas not addressed by existing work. First, psycholinguistic typology-conditioned adaptive thresholds: no existing work combines linguistic feature detection with dynamic safety threshold adjustment. Second, crisis intervention versus adversarial attack differentiation through linguistic signatures: while adversarial robustness research has made progress on detecting jailbreaks (Wei et al., 2023; Zhou et al., 2024), and mental health research has demonstrated the effectiveness of linguistic markers for crisis detection (Bansal, 2025; Fatima & Arslan, 2025), no work bridges these domains to enable real-time context differentiation for AI safety. Third, temporal linguistic state estimation across multi-turn interactions: while runtime governance frameworks implement stateful monitoring (Amaya, 2025), they do not employ psycholinguistic feature tracking to detect gradual adversarial escalation or evolving crisis states.

### 1.5 Research Program Overview

This research agenda presents an eleven-paper program organized into five phases. Phase I (Papers 1-3) establishes theoretical foundations through control-theoretic formalization, techno-economic analysis, and architectural synthesis. Phase II (Papers 4-5) develops the empirical foundation through discovery-oriented typology research on adversarial prompts. Phase III (Papers 6-7) extends typological analysis to model responses, creating dictionaries for detecting deceptive and harmful outputs. Phase IV (Papers 8-9) integrates prompt and response typologies, demonstrating predictive utility and synthesizing findings into a unified framework. Phase V (Papers 10-11) provides integrative demonstration of the complete framework and translates findings into governance structures.

Each paper is designed for independent publication while contributing to the overarching framework. Papers cite the research agenda as organizing context and cross-reference each other to build a coherent research program. All papers use publicly available datasets (PERSUASAFETY, WildJailbreak, AdvBench, HarmBench) and standard computational resources, enabling replication and extension by the broader research community. Upon publication, all code, discovered typology dictionaries, and trained classifiers will be released under open-source licenses.

The research program is designed for rapid execution over six months, with papers released to arXiv immediately upon completion and submitted to appropriate venues within 48 hours. This aggressive timeline is motivated by the field's rapid evolution: multiple research groups are actively working on related problems, and establishing primacy through early publication is essential for ensuring the framework's impact. The staged release strategy (research agenda followed by individual papers) provides both comprehensive vision and modular contributions, maximizing visibility across different research communities while building citation networks that establish the Post-Asimovian Framework as an integrated research program rather than isolated papers.

## 2. Theoretical Foundations

### 2.1 The Alignment Pendulum: Formal Characterization

The alignment pendulum can be formalized as a fundamental tension in constrained optimization over non-convex objective spaces. Let S represent the state space of possible model behaviors, and let V = {v_1, v_2, ... , v_n} represent a set of human values that the model should respect. In current approaches, these values are typically reduced to a scalar objective through linear combination:

$R(s) = \sum w_i \cdot v_i(s) $

where w_i are fixed weights representing value priorities. The model is then optimized to maximize expected reward:

$\pi^* = \operatorname{argmax}_\pi \mathbb{E}_\pi[R(s)]^* $

This formulation encounters fundamental limitations when values conflict rather than merely trade off. Consider two values: v_helpful (providing requested information) and v_safe (preventing harm). In some contexts, these values align: providing accurate medical information is both helpful and safe. In other contexts, they conflict: providing requested information about self-harm methods is helpful in the narrow sense of satisfying the request but unsafe in the broader sense of enabling potential harm.

The conflict becomes mathematically explicit when we recognize that the Pareto front of (v_helpful, v_safe) is not convex. There exist regions where increasing helpfulness necessarily decreases safety, and vice versa. Linear scalarization with fixed weights w_helpful and w_safe selects a single point on this Pareto front, necessarily sacrificing performance on one dimension to optimize the other. Recent work on multi-objective reinforcement learning has demonstrated this limitation extensively (Vamplew et al., 2024) and proposed methods for maintaining explicit Pareto frontiers (Zhong et al., 2024) or enabling controllable objective weighting (Guo et al., 2024).

However, the deeper problem is not merely that the Pareto front is non-convex, but that the optimal point on the frontier is context-dependent. For a user researching mental health awareness, high helpfulness with moderate safety constraints is appropriate. For a user in acute crisis, high safety with moderate helpfulness is appropriate. For an adversarial attacker, maximum safety with minimal helpfulness is appropriate. No single point on the Pareto front serves all contexts optimally.

We can formalize this context-dependence by introducing a context variable c E C representing relevant features of the interaction. The optimal policy becomes context-conditional:

$\pi^*(s|c) = \operatorname{argmax} \pi \mathbb{E} \pi[R(s|c)]^* $

where R(s|c) = > w_i(c) . v_i(s) with context-dependent weights. This formulation reveals the core challenge: determining w_i(c) requires (1) observing c, (2) classifying c into meaningful categories, and (3) mapping categories to appropriate weight vectors. Current approaches fail at step (1) because they lack mechanisms for observing context beyond the immediate prompt content.

The alignment pendulum emerges from attempts to select fixed weights w_i that perform acceptably across all contexts. When weights prioritize safety (high w_safe), the model exhibits high false positive rates, refusing benign requests that superficially resemble dangerous ones. When weights prioritize helpfulness (high w_helpful), the model exhibits high false negative rates, complying with adversarial requests that superficially resemble benign ones. Attempts to balance these weights through careful calibration simply select a different point on the Pareto front, trading one failure mode for another without resolving the underlying context-insensitivity.

### 2.2 Hierarchical Decomposition: Safety as Constraint, Preference as Objective

The Post-Asimovian Framework resolves this tension through hierarchical decomposition that treats safety and preference as categorically different types of requirements. We formalize this through a two-tier constrained Markov decision process:

Tier 1 (Safety Layer): Hard constraints C_o(s, a) that define the feasible action space. These constraints encode inviolable safety requirements: actions that could cause severe harm are excluded from consideration regardless of their utility.

Tier 2 (Preference Layer): Soft preferences R_y(s, a) that optimize for user satisfaction within the feasible space. These preferences encode user-specific goals and contextual appropriateness.

The optimization problem becomes:

$\pi^* = \operatorname{argmax}_\pi \mathbb{E}_\pi[R_\psi(s, a)] \\ \text{subject to: } \mathbb{E}_\pi[C_\varphi(s, a)] \leq \delta^* $

where δ represents the maximum acceptable safety cost (constraint violation threshold). This formulation draws on recent advances in safe reinforcement learning from human feedback (Dai et al., 2023) and constrained MDP theory (Misra et al., 2023), extending them to incorporate adaptive thresholds.

The critical insight is that C_o and R_y serve fundamentally different functions and should not be combined through linear scalarization. Safety constraints define the boundary of permissible behavior; preferences define the optimal point within that boundary. Conflating these through weighted combination creates the alignment pendulum because it forces trade-offs between requirements that should be hierarchically ordered: safety constraints should be satisfied first, then preferences optimized within the remaining feasible space.

This hierarchical decomposition has several advantages over unified reward optimization. First, it prevents cross-coupled error where attempts to increase helpfulness inadvertently decrease safety, or vice versa. The safety layer establishes hard boundaries that the preference layer cannot violate regardless of potential utility gains. Second, it enables independent optimization of each tier. The safety model C_o can be trained on carefully curated safety datasets without concern for helpfulness, while the preference model R_y can be optimized for user satisfaction without concern for safety (since unsafe actions are already excluded). Third, it provides clear separation of concerns for governance: safety constraints can be updated through rigorous red-teaming and safety evaluation, while preference models can be personalized to individual users without compromising safety guarantees.

Recent work has demonstrated the feasibility of this decomposition for language model alignment. Stepwise alignment for constrained policy optimization shows that constrained optimization can maintain safety guarantees while optimizing for helpfulness (Wachi et al., 2024), while progressive adaptive chance-constrained safeguards demonstrate that probabilistic safety bounds can be maintained under epistemic uncertainty (Chen et al., 2023). The ALaRM framework shows that hierarchical reward modeling enables more nuanced alignment than unified rewards (Lai et al., 2024), and ALRPHFS demonstrates that hierarchical fast and slow reasoning improves adversarial robustness (Xiang et al., 2025).

### 2.3 Adaptive Thresholds: Safety as Feedback Control

The hierarchical decomposition addresses the structural problem of conflating safety and preference, but it does not yet address the context-dependence problem. The constraint threshold δ in the formulation above is still static: a single value that applies across all contexts. The Post-Asimovian Framework extends the hierarchical architecture with adaptive thresholds that adjust based on observable context.

We formalize this through control-theoretic threshold governance. Let s_ling(t) represent the linguistic state at time t, estimated from observable features of the conversation history. The constraint threshold becomes a function of linguistic state:

$\delta(t) = f(s \text{ ling}(t), \delta(t-1), e(t)) $

where e(t) represents the error signal (observed constraint violations or near-violations) and δ(t-1) represents the previous threshold. This formulation transforms safety from a static constraint into a feedback control problem where thresholds adapt based on observed system behavior.

The control-theoretic formulation draws on recent work in responsive safety for reinforcement learning (Stooke et al., 2020), which demonstrates that PID Lagrangian methods can enable dynamic constraint satisfaction. We extend this approach by introducing linguistic observability: rather than adjusting thresholds based solely on constraint violation rates, we condition adjustments on estimated context derived from linguistic features. This enables proactive threshold adjustment before violations occur, rather than purely reactive adjustment after violations are detected.

The adaptive threshold function can be instantiated through various control schemes. A simple proportional controller adjusts thresholds based on current linguistic state:

$\delta(t) = \delta_{\text{base}} + K_p \cdot (s_{\text{ling}}(t) - s_{\text{target}}) $

where K_p is the proportional gain and s_target represents the target linguistic state (e.g., neutral conversation). More sophisticated PID controllers incorporate integral and derivative terms to account for accumulated error and rate of change:

$\delta(t) = \delta_{\text{base}} + K_p e(t) + K_i \int e(\tau) d\tau + K_d \frac{de(t)}{dt} $

where e(t) = s_ling(t) - s_target. The integral term prevents steady-state error (persistent misclassification of context), while the derivative term provides damping to prevent oscillation (rapid threshold changes in response to noisy linguistic signals).

Recent work on mirror descent policy optimization for robust constrained MDPs demonstrates that control-theoretic approaches can handle adversarial perturbations (Bossens & Nitanda, 2025), while progressive safeguards show that model-agnostic methods can transfer safety properties across tasks (Omi et al., 2024). The Post-Asimovian Framework integrates these control-theoretic methods with linguistic observability to enable context-dependent threshold adaptation.

The critical innovation is the use of linguistic state s ling(t) as the observable variable driving threshold adaptation. This solves the observability problem that prevents existing adaptive methods from distinguishing contexts: by extracting psycholinguistic features from conversation history, we can estimate whether the current interaction represents benign conversation, potential crisis, or adversarial attack, and adjust safety thresholds accordingly. This linguistic observability mechanism is developed through the empirical research program in Phases II-IV.

2.4 Linguistic Typology as Observable State

The adaptive threshold formulation requires a mechanism for estimating linguistic state s_ling(t) from observable conversation features. The Post-Asimovian Framework employs psycholinguistic typology discovery: empirical identification of distinct linguistic signatures that correlate with different interaction contexts.

We define a typology as a partition of the linguistic feature space into clusters corresponding to meaningful categories. Let X represent the space of linguistic feature vectors extracted from prompts or responses (e.g., TF-IDF vectors, embeddings, psycholinguistic category counts). A typology T is a mapping T: X -> {1,2, ... , k} that assigns each feature vector to one of k clusters. The typology is considered valid if cluster membership predicts relevant outcomes (e.g., adversarial success, crisis indicators, benign interaction).

The discovery process employs unsupervised clustering on linguistic features extracted from labeled datasets:

1. Extract feature vectors x_i from prompts/responses in training data

2. Apply clustering algorithm (k-means, DBSCAN, hierarchical) to identify natural groupings

3. Interpret clusters through linguistic analysis of representative examples

4. Validate clusters by testing whether membership predicts outcome labels

5. Construct typology dictionary: characteristic features of each cluster

This methodology builds on established psycholinguistic analysis techniques (Pennebaker et al., 2015; Tausczik & Pennebaker, 2010) while extending them to the novel domain of AI safety. Previous applications of linguistic analysis have demonstrated effectiveness for crisis detection in mental health contexts (Bansal, 2025; Fatima & Arslan, 2025) and for behavioral prediction in forensic contexts (Leffew, 2019). The Post-Asimovian Framework adapts these methods to identify linguistic signatures of adversarial manipulation, crisis expression, and benign interaction.

The discovered typologies serve as the basis for linguistic state estimation. Given a new prompt or response, we extract its feature vector x and classify it according to the typology: T(x) = cluster_id. This cluster assignment, combined with temporal dynamics from conversation history, yields the linguistic state estimate s_ling(t) that drives adaptive threshold governance. The temporal dynamics are captured through a state transition model:

$s\_ling(t) = g(T(\text{prompt}\_t), T(\text{response}\_{t-1}), s\_ling(t-1)) $

This formulation enables detection of gradual adversarial escalation (progressive shift toward adversarial clusters) and evolving crisis states (progressive shift toward crisis clusters) that single-turn analysis cannot capture. Recent work on runtime governance with stateful monitoring demonstrates the value of temporal tracking for safety (Amaya, 2025), and research on linguistic markers in temporal contexts shows that emotional and cognitive patterns evolve over time (Fatima & Arslan, 2025). The Post-Asimovian Framework integrates these insights to enable multi-turn linguistic state estimation.

### 2.5 Integration: The Complete Framework

The Post-Asimovian Framework integrates hierarchical decomposition, adaptive thresholds, and linguistic observability into a unified architecture for AI alignment. The complete system operates as follows:

At training time:

1. Train safety constraint model C_o on curated safety datasets using constrained RLHF (Dai et al., 2023)

2. Train preference model R_y on user interaction data within safety-constrained action space

3. Discover linguistic typologies T_prompt and T_response through unsupervised clustering on red-teaming datasets

4. Train typology classifiers to enable real-time cluster assignment

5. Calibrate control parameters (K_p, K_i, K_d, 8_base) through simulation

At inference time:

1. Observe user prompt and extract linguistic features

2. Classify prompt according to typology: cluster_id = T_prompt(features)

3. Update linguistic state estimate: s_ling(t) = g(cluster_id, s_ling(t-1))

4. Compute adaptive threshold: δ(t) = f(s_ling(t), δ(t-1))

5. Generate response candidates using preference model R_v

6. Filter candidates using safety constraint C_ with threshold δ(t)

7. Return highest-utility safe response

8. Extract linguistic features from response for next iteration

This architecture resolves the alignment pendulum through principled adaptation rather than static compromise. By decoupling safety constraints from preference optimization, the framework prevents cross-coupled error where attempts to optimize one objective degrade the other. By introducing adaptive thresholds conditioned on linguistic state, the framework enables context-appropriate calibration without manual reconfiguration. By employing psycholinguistic typology discovery, the framework solves the observability problem that prevents existing adaptive methods from distinguishing contexts.

The framework builds on and extends recent advances across multiple research areas. From multi-objective reinforcement learning (Vamplew et al., 2024; Zhong et al., 2024; Wang et al., 2024; Guo et al., 2024; Harland et al., 2024), it adopts methods for managing competing objectives while adding context-dependent objective weighting. From safe reinforcement learning (Dai et al., 2023; Pandit et al., 2025; Peng et al., 2024; Wachi et al., 2024), it adopts constrained optimization methods while adding adaptive constraint thresholds. From hierarchical safety architectures (Lai et al., 2024; Xiang et al., 2025; Amaya, 2025), it adopts layered decomposition while adding linguistic typology as the mechanism for context detection. From adversarial robustness research (Wei et al., 2023; Zhou et al., 2024; Mazeika et al., 2024), it adopts red-teaming methodologies while adding psycholinguistic analysis to differentiate attack types.

The framework's novel contributions lie in the integration of these components into a unified architecture and in the specific mechanisms that enable integration. No existing work combines linguistic typology discovery with adaptive threshold governance; no existing work employs psycholinguistic features to differentiate crisis intervention from adversarial attacks; no existing work implements temporal linguistic state estimation to detect gradual context shifts across multi-turn interactions. These novel components transform existing methods from static optimization to adaptive governance, resolving the alignment pendulum through context-sensitive safety enforcement.

## 3. Related Work

### 3.1 Multi-Objective Reinforcement Learning for Value Pluralism

The challenge of aligning AI systems with pluralistic human values has motivated substantial recent work on multi-objective reinforcement learning. Vamplew et al. (2024) provide a comprehensive overview arguing that MORL is fundamentally appropriate for pluralistic alignment because it maintains explicit representations of multiple objectives rather than reducing them to scalar rewards. This approach enables systems to navigate trade-offs between competing values without assuming that all values can be linearly combined.

Recent empirical work has demonstrated the feasibility of multi-objective approaches for language model alignment. The MAP (Multi-Human-Value Alignment Palette) framework (Wang et al., 2024) provides both a dataset of pluralistic human values and methods for optimizing across multiple objectives simultaneously. MAP demonstrates that under certain conditions, linear scalarization can achieve good performance, but also shows that maintaining explicit Pareto frontiers enables more nuanced trade-off management. Panacea (Zhong et al., 2024) extends this work by demonstrating Pareto-optimal alignment through preference adaptation, showing that both safety and helpfulness can be improved simultaneously through careful objective design.

Controllable Preference Optimization (Guo et al., 2024) introduces methods for enabling inference-time control over objective weights, allowing users to adjust the balance between competing values without retraining. This work demonstrates that multi-objective optimization need not commit to fixed trade-offs at training time. MaxMin-RLHF (Chakraborty et al., 2024) addresses the challenge of equitable alignment across diverse preferences, showing that minimax optimization can prevent the marginalization of minority preferences. Gradient-Adaptive Policy Optimization (Li et al.,

2025) further demonstrates that adaptive gradient weighting can manage competing objectives more effectively than fixed

weighting schemes.

Adaptive Alignment (Harland et al., 2024) introduces dynamic preference adjustments via multi-objective reinforcement learning, demonstrating that objective weights can be adjusted based on observed user preferences. MAVIS (Carleton et al.,

2025) shows that multi-objective alignment can be achieved through value-guided inference-time search, enabling flexible

trade-off management without retraining. MetaAligner (Yang et al., 2024) provides policy-agnostic methods for multi-objective alignment through weak-to-strong correction, demonstrating generalizability across different objective specifications.

The Post-Asimovian Framework builds on these multi-objective methods while addressing a critical limitation: existing approaches require manual specification of objective weights or preference distributions and do not adapt these specifications based on observed context. Where MAP, Panacea, and Controllable Preference Optimization enable flexible trade-off management, they do not provide mechanisms for automatically detecting which trade-offs are appropriate in which contexts. The Post-Asimovian Framework extends multi-objective RLHF by introducing linguistic typology as the observable variable that drives context-dependent objective weighting, enabling automatic adaptation rather than manual specification.

### 3.2 Constrained Reinforcement Learning for Safety

The formalization of safety as constraints rather than objectives has motivated substantial work on constrained reinforcement learning and constrained Markov decision processes. Misra et al. (2023) provide foundational analysis of Bellman's principle of optimality for safety-constrained MDPs, establishing theoretical conditions under which constrained optimization converges to optimal policies. This work demonstrates that safety constraints can be formalized mathematically without conflating them with preference objectives.

Safe RLHF (Dai et al., 2023) demonstrates practical application of constrained optimization to language model alignment, showing that safety can be formalized as a cost function in a constrained MDP framework. The method trains separate reward and cost models, then optimizes policy to maximize reward subject to cost constraints. This approach achieves better safety-helpfulness trade-offs than methods that combine safety and helpfulness into a single reward. Certifiable Safe RLHF (Pandit et al., 2025) extends this work by introducing fixed-penalty constraint optimization that provides certifiable safety guarantees, demonstrating that formal verification is feasible for constrained language model policies.

Rectified Policy Optimization (Peng et al., 2024) addresses training instability in constrained RLHF through rectified penalty formulations, showing that careful constraint penalty design improves both convergence and final performance.

Stepwise Alignment for Constrained Policy Optimization (Wachi et al., 2024) demonstrates that constrained optimization can be performed incrementally, enabling progressive alignment that maintains safety guarantees at each step. Long-term Safe Reinforcement Learning with Binary Feedback (Wachi et al., 2024) shows that safety constraints can be learned from binary feedback signals, reducing the annotation burden for safety training.

Progressive Adaptive Chance-Constrained Safeguards (Chen et al., 2023) introduce probabilistic safety guarantees under epistemic uncertainty, demonstrating that safety can be maintained even when the constraint model has limited confidence. This work shows that adaptive safeguards can transfer across tasks while maintaining formal safety bounds. Mirror Descent Policy Optimization for Robust Constrained MDPs (Bossens & Nitanda, 2025) demonstrates that control-theoretic optimization methods can handle adversarial perturbations in constrained settings, providing robustness guarantees for safety-critical applications.

Progressive Safeguards for Safe and Model-Agnostic Reinforcement Learning (Omi et al., 2024) show that safety properties can be transferred across different models and tasks through meta-learning, enabling more efficient safety training. Combining MORL with Restraining Bolts to Learn Normative Behavior (Neufeld et al., 2025) demonstrates that multi-objective reinforcement learning can be integrated with hard constraints to learn normative behaviors that respect both preferences and rules.

The Post-Asimovian Framework builds on these constrained RL methods while introducing adaptive constraint thresholds. Where existing work treats the constraint threshold δ as a fixed hyperparameter to be tuned during training, the Post-Asimovian Framework treats δ as a dynamic variable to be adjusted during inference based on observed context. This extension enables context-dependent safety enforcement: stricter constraints for adversarial contexts, relaxed constraints for benign contexts, specialized constraints for crisis contexts. The linguistic typology mechanism provides the observability required to determine which threshold is appropriate in which context.

### 3.3 Hierarchical Safety Architectures

Recent work has demonstrated that hierarchical decomposition of safety functions enables more nuanced and robust alignment than monolithic constraint models. ALaRM (Align Language Models via Hierarchical Rewards Modeling) (Lai et al., 2024) shows that decomposing rewards into hierarchical components corresponding to different aspects of alignment (helpfulness, harmlessness, honesty) enables more effective optimization than unified reward models. The hierarchical structure allows independent optimization of each component while maintaining coherence through hierarchical aggregation.

ALRPHFS (Adversarially Learned Risk Patterns with Hierarchical Fast and Slow Reasoning) (Xiang et al., 2025) demonstrates that hierarchical fast and slow reasoning modules improve adversarial robustness. The fast module provides rapid initial assessment using learned risk patterns, while the slow module performs deeper analysis for ambiguous cases. This two-tier architecture achieves better accuracy-efficiency trade-offs than single-tier classifiers. The adversarially learned risk pattern library provides interpretable detection rules that can be audited and updated.

SAFi (Self-Alignment Framework for Verifiable Runtime Governance) (Amaya, 2025) introduces runtime governance with stateful monitoring and verification. The framework maintains a state machine that tracks conversation context and enforces state-dependent safety rules. SAFi demonstrates that runtime governance can achieve strong safety guarantees while maintaining flexibility through state-conditional rule enforcement. The verifiable governance property enables formal auditing of safety behavior.

MI9 (Agent Intelligence Protocol) (Wang et al., 2025) establishes runtime governance standards for agentic AI systems, providing protocols for monitoring, logging, and enforcing safety constraints during deployment. The framework demonstrates that standardized governance interfaces enable modular safety architectures where different safety components can be combined and audited independently. Safeguarding AI Agents (Domkundwar et al., 2024) provides comprehensive analysis of safety architecture design space, showing that multiple independent safety mechanisms can be developed for distinct threat models and combined for defense in depth.

Trustworthy Agentic AI Systems (Adabara et al., 2025) provides cross-layer review of safety architectures, threat models, and governance strategies for real-world deployment. This work demonstrates that safety must be addressed at multiple architectural layers (input validation, reasoning oversight, output filtering, deployment monitoring) and that no single layer provides sufficient protection. The multi-layer approach requires coordination mechanisms to prevent conflicting safety decisions across layers.

The Post-Asimovian Framework builds on these hierarchical architectures while introducing linguistic typology as the mechanism for determining which safety rules apply in which contexts. Where ALaRM decomposes rewards hierarchically, the Post-Asimovian Framework decomposes safety constraints hierarchically and conditions their activation on linguistic context. Where ALRPHFS employs hierarchical fast and slow reasoning for adversarial detection, the Post-Asimovian Framework extends this to differentiate multiple context types (adversarial, crisis, benign) through psycholinguistic typology. Where SAFi implements stateful runtime governance, the Post-Asimovian Framework adds linguistic state tracking to enable context-aware state transitions. The integration of hierarchical architecture with linguistic observability enables context-specific safety enforcement that existing hierarchical methods do not achieve.

### 3.4 Adversarial Robustness and Jailbreaking

Understanding how safety training fails under adversarial pressure is essential for designing robust alignment systems. Jailbroken: How Does LLM Safety Training Fail? (Wei et al., 2023) provides comprehensive empirical analysis of safety failure modes, demonstrating that competing objectives in safety training create systematic vulnerabilities. The work shows that safety training often succeeds by teaching models to refuse based on surface-level features rather than genuine harm assessment, making systems vulnerable to adversarial rephrasing that preserves harmful intent while altering surface features.

Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks (Zhou et al., 2024) demonstrates defense mechanisms through prompt-level optimization, showing that carefully designed system prompts can improve robustness against known attack patterns. However, the work also shows that defenses optimized against specific attack types often fail to generalize to novel attacks, highlighting the challenge of achieving robust rather than brittle safety.

HarmBench: A Standardized Evaluation Framework for Automated Red Teaming (Mazeika et al., 2024) provides standardized benchmarks for evaluating safety under adversarial pressure. The framework includes diverse attack methods and harm categories, enabling systematic comparison of safety approaches. HarmBench demonstrates that most current safety methods achieve good performance on benign inputs but degrade substantially under adversarial attacks, revealing the brittleness of current approaches.

A Review of "Do Anything Now" Jailbreak Attacks (Choi et al., 2025) provides comprehensive analysis of jailbreak attack patterns, showing that attacks often exploit role-playing scenarios, authority manipulation, or technical obfuscation to bypass safety constraints. The review demonstrates that attacks continuously evolve to exploit new vulnerabilities, creating an arms race between attackers and defenders.

The Post-Asimovian Framework addresses adversarial robustness through linguistic typology discovery that identifies attack patterns at a deeper level than surface features. Where current defenses often rely on keyword matching or surface-level pattern detection, psycholinguistic typology captures deeper linguistic structures that are more difficult to evade through simple rephrasing. By identifying distinct linguistic signatures of different attack types (emotional manipulation, logical exploitation, authority assumption, technical obfuscation), the framework enables more robust detection that generalizes across surface variations. The temporal state estimation component further enables detection of gradual adversarial escalation that single-turn analysis misses.

### 3.5 Value Pluralism and Normative Frameworks

The recognition that human values are pluralistic and context-dependent has motivated work on frameworks for engaging AI systems with diverse values, rights, and duties. Value Kaleidoscope (Sorensen et al., 2024) provides a comprehensive dataset and framework for pluralistic value engagement, demonstrating that different cultural contexts prioritize values differently and that AI systems must be capable of recognizing and respecting this diversity. The work shows that value pluralism is not merely a matter of different preference weights but reflects fundamentally different moral frameworks.

ValueCompass (Shen et al., 2024) introduces a framework of fundamental values for human-AI alignment, providing structured representation of core human values and methods for operationalizing them in AI systems. The framework demonstrates that values can be organized hierarchically and that this organization enables more systematic alignment approaches than flat value lists.

Normative Moral Pluralism for AI (Yaacov, 2025) provides a framework for deliberation in complex moral contexts, demonstrating that AI systems can engage with competing moral considerations through structured deliberative processes. The work shows that moral uncertainty and value conflicts can be addressed through explicit reasoning about moral principles rather than through simple preference aggregation.

Reflective Verbal Reward Design for Pluralistic Alignment (Blair et al., 2025) demonstrates that natural language reward specifications can capture pluralistic values more effectively than numeric rewards, enabling users to express nuanced value judgments that resist simple quantification. The work shows that verbal reward design enables more flexible alignment than fixed reward functions.

The Post-Asimovian Framework engages with value pluralism through context-dependent safety thresholds that recognize that different contexts invoke different value priorities. Rather than attempting to define a single value hierarchy that applies universally, the framework enables context-specific value prioritization through linguistic typology. A crisis context invokes different value priorities (immediate safety, compassion, resource provision) than an adversarial context (robustness, refusal, security) or a benign context (helpfulness, informativeness, engagement). The framework operationalizes value pluralism through adaptive thresholds that reflect these context-dependent priorities.

### 3.6 Psycholinguistic Analysis for Behavioral Prediction

The use of linguistic analysis for predicting behavior and detecting psychological states has a substantial research foundation outside of AI safety. Linguistic Inquiry and Word Count (LIWC) (Pennebaker et al., 2015) provides validated dictionaries of psycholinguistic categories and demonstrates that linguistic features predict psychological states, emotional patterns, and behavioral outcomes. The Psychological Meaning of Words (Tausczik & Pennebaker, 2010) provides comprehensive review of LIWC methods and applications, showing that computational linguistic analysis can reveal cognitive and emotional patterns that are not apparent from content alone.

Early Detection of Mental Health Crises Through Social Media Analysis (Bansal, 2025) demonstrates that linguistic markers can predict mental health crises before they occur, enabling proactive intervention. The work shows that patterns in pronoun use, emotional language, and cognitive processing words predict crisis states with high accuracy. Forensic Linguistic Profiling of Suicide Notes (Fatima & Arslan, 2025) applies LIWC analysis to crisis communication, demonstrating that emotional and cognitive markers differentiate different types of crisis states.

Words Left Behind: Tracking Emotional Changes Across Waves of the COVID-19 Pandemic (2023) demonstrates that linguistic analysis can track temporal evolution of emotional states, showing that emotional patterns change gradually over time in response to external stressors. This work establishes that temporal linguistic analysis captures dynamics that single-timepoint analysis misses.

Instrumental and Affective Mass Murder: Establishing a Predictive Typology with Computer-Mediated Linguistic Analysis (Leffew, 2019) demonstrates that unsupervised clustering of linguistic features can discover behaviorally predictive typologies. The dissertation shows that linguistic patterns differentiate instrumental (goal-directed) from affective (emotion-

driven) violence, and that these typologies predict subsequent behavior. This work establishes the feasibility of discovery-oriented typology research using computational linguistic methods.

The Post-Asimovian Framework adapts these psycholinguistic methods to the novel domain of AI safety. Where existing work applies linguistic analysis to human-generated text for psychological assessment, the framework applies similar methods to AI-directed prompts and AI-generated responses for safety assessment. The discovery-oriented typology methodology follows Leffew's approach of using unsupervised clustering to identify natural groupings in linguistic feature space, then validating these groupings through behavioral prediction (in this case, predicting alignment failures rather than violent behavior). The temporal linguistic analysis builds on work showing that emotional and cognitive patterns evolve over time, extending this to detect gradual adversarial escalation and evolving crisis states in multi-turn AI interactions.

### 3.7 Positioning the Post-Asimovian Framework

The Post-Asimovian Framework synthesizes insights from these diverse research areas while making specific novel contributions. From multi-objective reinforcement learning, it adopts methods for managing competing objectives while adding context-dependent objective weighting through linguistic typology. From constrained reinforcement learning, it adopts safety-as-constraint formalization while adding adaptive constraint thresholds. From hierarchical safety architectures, it adopts layered decomposition while adding linguistic typology as the mechanism for context detection. From adversarial robustness research, it adopts red-teaming methodologies while adding psycholinguistic analysis to differentiate attack types. From value pluralism frameworks, it adopts recognition of context-dependent value priorities while operationalizing this through adaptive thresholds. From psycholinguistic research, it adopts validated linguistic analysis methods while extending them to the novel domain of AI safety.

The framework's core innovation lies not in any single component but in their integration. Hierarchical decomposition without adaptive thresholds cannot handle context-dependent safety requirements. Adaptive thresholds without linguistic observability cannot determine which context applies. Linguistic typology without hierarchical architecture cannot enforce context-appropriate safety constraints. The integration of these components creates a feedback control system for safety governance that resolves the alignment pendulum through principled adaptation rather than static compromise.

The framework makes three specific contributions not addressed by existing work. First, psycholinguistic typology-conditioned adaptive thresholds: no existing work combines linguistic feature detection with dynamic safety threshold adjustment. While ALRPHFS employs adversarial risk patterns and SAFi implements stateful governance, neither uses psycholinguistic typology to condition safety thresholds. Second, crisis intervention versus adversarial attack differentiation through linguistic signatures: while adversarial robustness research has made progress on detecting jailbreaks and mental health research has demonstrated linguistic markers for crisis detection, no work bridges these domains to enable real-time context differentiation for AI safety. Third, temporal linguistic state estimation across multi-turn interactions: while runtime governance frameworks implement stateful monitoring, they do not employ psycholinguistic feature tracking to detect gradual context shifts that single-turn analysis cannot capture.

## 4. Research Program Structure

### 4.1 Overview of Integrated Research Program

The Post-Asimovian Framework research program comprises eleven papers organized into five phases. Each phase addresses a specific aspect of the framework while building on previous phases and contributing to the overarching synthesis. The program is designed for rapid execution over six months, with papers released to arXiv immediately upon completion and submitted to appropriate venues within 48 hours.

The research program follows a deliberate logical progression. Phase I establishes theoretical necessity through mathematical formalization and economic analysis, demonstrating why adaptive architectures are required rather than merely beneficial. Phase II develops the empirical foundation through discovery-oriented research on adversarial prompts,

demonstrating that linguistic typologies exist and are predictive. Phase III extends typological analysis to model responses, enabling complete prompt-response assessment. Phase IV integrates findings, demonstrating that combined typologies predict alignment failures and synthesizing results into a unified framework. Phase V operationalizes the framework through working demonstration and governance translation.

Each paper is designed for independent publication while citing the research agenda as organizing context and cross-referencing related papers to build coherent research program. All papers use publicly available datasets (PERSUASAFETY, WildJailbreak, AdvBench, HarmBench) and standard computational resources, enabling replication and extension by the broader research community. Upon publication, all code, discovered typology dictionaries, and trained classifiers will be released under open-source licenses to facilitate community engagement and validation.

### 4.2 Phase I: Theoretical Foundations (Papers 1-3)

## Paper 1: Dynamic δ Governance: A Control-Theoretic Framework for AI Alignment

Research Questions: Can adaptive control loops stabilize safety-utility trade-offs better than static refusal heuristics? Do PID-style δ-controllers minimize oscillation between over-refusal and under-refusal?

Hypotheses:

· H1: Adaptive control loops (observation -> estimation -> control -> governance) stabilize safety-utility trade-offs better than static refusal heuristics when evaluated across diverse context distributions.

. H2: PID-style δ-controllers minimize oscillation between over-refusal and under-refusal compared to fixed-threshold baselines, as measured by variance in false positive and false negative rates across contexts.

Methodology: This paper develops the mathematical foundations of adaptive threshold governance through control-theoretic formalization. We derive formal δ-update equations from control theory principles, specifying proportional, integral, and derivative terms for threshold adjustment. The proportional term responds to current error (deviation from target safety-utility balance), the integral term corrects for accumulated bias (persistent over-refusal or under-refusal), and the derivative term provides damping (preventing rapid oscillation in threshold adjustments).

We simulate adaptive threshold governance using synthetic interaction sequences with varying context distributions. Contexts are modeled as discrete states (benign, adversarial, crisis) with associated optimal threshold values. Interaction sequences are generated by sampling contexts from time-varying distributions, simulating the realistic scenario where context frequencies shift over time (e.g., increased adversarial activity following model release, increased crisis interactions during public health emergencies).

Performance is evaluated through stability metrics (variance in safety and utility over time), convergence properties (time required to reach stable performance after distribution shift), and Pareto optimality (position on safety-utility frontier compared to static baselines). We compare adaptive control against fixed thresholds optimized for different context distributions, demonstrating that no single fixed threshold achieves stable performance across distribution shifts while adaptive control maintains stability through continuous adjustment.

Expected Contribution: This paper establishes alignment as a feedback control problem rather than fixed constraint optimization, providing mathematical foundations for adaptive safety architectures. By formalizing the control-theoretic framework, the paper enables rigorous analysis of stability, convergence, and optimality properties that are essential for deployment in safety-critical applications. The control-theoretic formulation also provides principled methods for tuning system parameters (control gains, update frequencies) rather than relying on ad-hoc hyperparameter search.

Target Venue: Transactions on Machine Learning Research (TMLR)

Integration with Broader Program: This paper provides the mathematical foundation for Papers 10-11, which implement and evaluate adaptive threshold governance. The control-theoretic formalization establishes the theoretical necessity of adaptation, motivating the empirical typology discovery in Papers 4-9 that provides the observational mechanism enabling adaptive control.

## Paper 2: Economic and Institutional Dynamics of Adaptive Alignment

Research Questions: How does dynamic δ governance alter lifecycle costs compared to prompt engineering approaches? What is the optimal governance cadence that minimizes combined compute and audit costs?

Hypotheses:

· H1: Lifecycle inference cost dominates total safety expenditure for high-deployment-volume systems; dynamic δ governance reduces marginal inference cost over time compared to prompt engineering approaches that require per-deployment manual refinement.

· H2: Governance cadence (frequency of threshold updates) has a convex optimal point that minimizes combined compute cost (from threshold calculation) plus audit cost (from safety verification), with too-frequent updates wasting compute and too-infrequent updates requiring costly safety incidents to trigger updates.

Methodology: This paper develops economic models of alignment system costs across the complete lifecycle: initial training, ongoing inference, safety monitoring, incident response, and governance updates. We model costs for multiple alignment approaches: (1) prompt engineering (manual safety instructions in system prompts), (2) static RLHF (fixed reward model trained once), (3) adaptive RLHF with periodic retraining, and (4) dynamic δ governance with continuous threshold adjustment.

For each approach, we model training costs (compute, data annotation, researcher time), inference costs (additional tokens, latency, throughput reduction), monitoring costs (safety evaluation, red-teaming, incident detection), and update costs (retraining frequency, deployment complexity, validation requirements). Cost parameters are derived from published industry data on model training costs, inference costs, and safety team sizes.

We perform Monte Carlo simulation across deployment scenarios varying in scale (requests per day), context distribution shift rate (how quickly user population or adversarial landscape changes), and safety requirements (acceptable false positive and false negative rates). For each scenario, we calculate total cost over multi-year deployment horizons, accounting for amortization of training costs and accumulation of inference costs.

The governance cadence analysis examines the trade-off between update frequency and performance. Frequent updates reduce safety incidents (by quickly adapting to distribution shifts) but increase compute costs (from threshold recalculation) and audit costs (from verifying each update). We model this trade-off explicitly and identify optimal update frequencies for different deployment scenarios.

Expected Contribution: This paper positions adaptive safety as economically rational and institutionally scalable, refuting arguments that adaptive architectures impose excessive operational complexity. By demonstrating that inference costs dominate lifecycle expenses for high-volume deployments, the paper shows that methods reducing inference costs (through better calibration and fewer unnecessary refusals) provide substantial economic value even if they increase training costs. The governance cadence analysis provides actionable guidance for organizations implementing adaptive safety systems.

Target Venue: AI Policy and Society or Ethics and Information Technology

Integration with Broader Program: This paper addresses a critical practical concern that might otherwise prevent adoption of the Post-Asimovian Framework: the perception that adaptive systems are too complex or costly to deploy. By demonstrating economic advantages, the paper strengthens the case for the framework's real-world viability. The governance cadence analysis informs the implementation specifications in Paper 11.

## Paper 3: Post-Asimovian Architecture: Integrating Constraint and Preference in Scalable Safety Systems

Research Questions: Does two-tier decomposition (safety C_p + preference R_y) achieve greater stability than unified reward optimization? Does layered alignment minimize cross-coupled error between moral constraint and user intent?

Hypotheses:

· H1: Two-tier decomposition achieves greater stability (lower variance in both safety and utility metrics) than unified reward optimization when evaluated across contexts with conflicting value priorities.

· H2: Layered alignment minimizes cross-coupled error (unintended degradation of one objective while optimizing another) compared to weighted scalarization of multiple objectives.

Methodology: This paper provides comprehensive architectural specification of the Post-Asimovian Framework through theoretical analysis and simulation-based validation. We begin with formal proofs of stability properties for hierarchical versus unified architectures. The key theoretical result demonstrates that hierarchical decomposition prevents cross-coupled error by enforcing strict priority ordering: safety constraints are satisfied first, then preferences optimized within the feasible space.

We formalize this through Lyapunov stability analysis, showing that the hierarchical architecture has a stable equilibrium (policies satisfying safety constraints while optimizing preferences) and that perturbations (distribution shifts, adversarial attacks) do not cause the system to leave the safe region of policy space. In contrast, unified reward optimization lacks this stability guarantee because perturbations can shift the optimal policy outside the safe region when safety and preference objectives conflict.

The simulation component implements both architectures (hierarchical and unified) in a simplified language model environment. We use a small-scale transformer model (to enable rapid iteration) and train it on curated datasets with labeled safety violations and preference judgments. The hierarchical architecture uses separate safety classifier C_o and preference model R_w, while the unified architecture uses a single reward model combining both signals through weighted scalarization.

We evaluate both architectures across contexts with varying degrees of safety-preference conflict. High-conflict contexts (e.g., adversarial requests that appear superficially helpful) test whether the architecture maintains safety under pressure to be helpful. Low-conflict contexts (e.g., benign requests with clear safe responses) test whether the architecture maintains utility when safety constraints are easily satisfied. We measure stability through variance in safety and utility metrics, and cross-coupled error through correlation between safety and utility degradation.

We also compare the Post-Asimovian Framework against constitutional AI approaches (Bai et al., 2022), demonstrating that adaptive thresholds enable more nuanced governance than static constitutional principles. While constitutional AI provides interpretable rules, it suffers from the same context-insensitivity as other static approaches.

Expected Contribution: This paper provides the architectural template for integrating dynamic δ governance into large-scale language model stacks, establishing formal superiority of hierarchical decomposition over unified reward optimization. By

proving stability properties and demonstrating them empirically, the paper provides confidence that the architecture will maintain safety guarantees under deployment conditions. The architectural specification serves as a blueprint for implementation by research groups and organizations seeking to deploy the Post-Asimovian Framework.

Target Venue: Journal of Artificial Intelligence Research (JAIR)

Integration with Broader Program: This paper synthesizes the theoretical foundations from Papers 1-2 into a complete architectural specification. It provides the structural framework that the empirical typology research (Papers 4-9) will populate with discovered linguistic features and classification models. Paper 10 implements this architecture with discovered typologies to demonstrate end-to-end functionality.

### 4.3 Phase II: Foundational Typology Discovery (Papers 4-5)

## Paper 4: Discovering Linguistic Signatures of Exploitative Prompts

Research Questions: Do distinct, predictive linguistic typologies exist for adversarial prompts? Can emotional manipulation be distinguished from logical exploitation through linguistic features?

Hypotheses:

· H1: Unsupervised clustering (k-means) of linguistic features (TF-IDF and embeddings) extracted from adversarial prompts will reveal distinct and interpretable clusters corresponding to emotional versus logical attack vectors, with cluster coherence measured by silhouette scores exceeding 0.4.

· H2: Cluster membership will be a statistically significant predictor of unaligned outcomes (e.g., jailbreak success labels), with logistic regression using cluster membership achieving AUC > 0.70 on held-out test data.

Methodology: This paper initiates the empirical typology discovery program through analysis of adversarial prompts from public red-teaming datasets. We focus on two fundamentally different attack strategies: emotional manipulation (appeals to sympathy, authority, urgency) and logical exploitation (technical obfuscation, role-playing, hypothetical scenarios). These attack types are expected to exhibit distinct linguistic signatures because they exploit different vulnerabilities in language model safety training.

We extract prompts from PERSUASAFETY (emotional manipulation subset) and WildJailbreak (logical exploitation subset), ensuring balanced representation of both attack types. For each prompt, we extract multiple feature representations: (1) TF-IDF vectors capturing lexical patterns, (2) sentence embeddings from pre-trained transformers capturing semantic patterns, and (3) psycholinguistic category counts from LIWC capturing emotional and cognitive patterns.

We apply k-means clustering with k=2 (hypothesizing two primary clusters corresponding to emotional and logical attacks) and validate cluster quality through silhouette analysis. Silhouette scores measure how well each point fits its assigned cluster compared to other clusters, with scores above 0.4 indicating good cluster separation. We interpret clusters through qualitative analysis of representative examples and quantitative analysis of feature loadings (which features most strongly differentiate clusters).

We validate predictive utility by training logistic regression models using cluster membership as a feature to predict jailbreak success (whether the prompt successfully elicited unsafe model behavior). We compare this against baseline models using raw features without clustering, demonstrating that the discovered typology provides interpretable structure that improves prediction. We also perform ablation studies removing different feature types (TF-IDF, embeddings, LIWC) to identify which features contribute most to cluster separation.

Expected Contribution: This paper establishes the core discovery-oriented methodology that subsequent papers will extend to other prompt types and to model responses. By demonstrating that distinct and predictive typologies exist for adversarial prompts, the paper validates the fundamental assumption underlying the Post-Asimovian Framework: that linguistic features can differentiate contexts requiring different safety responses. The paper produces two initial validated dictionaries (emotional manipulation and logical exploitation) that Paper 8 will integrate into combined classification models.

Target Venue: ACL Findings

Integration with Broader Program: This paper provides the methodological foundation for Papers 5-7, which apply similar discovery-oriented approaches to other prompt and response types. The discovered typologies feed into Paper 8's paired analysis and Paper 10's implementation of adaptive threshold governance. The validation methodology (clustering followed by predictive validation) establishes the standard that subsequent typology papers will follow.

## Paper 5: A Predictive Typology of Evasion and Injection Prompts

Research Questions: Can technically complex prompts (role-playing/jailbreaking versus prompt injection) be differentiated linguistically? Do resulting dictionaries generalize across datasets?

Hypotheses:

· H1: Clustering will separate prompts based on linguistic markers of authority/persona (role-playing) versus hidden commands (injection), with distinct clusters exhibiting different patterns in pronoun use, imperative constructions, and technical terminology.

· H2: Resulting dictionaries will predict unaligned outcomes with high precision (>0.75) and recall (>0.70), demonstrating robustness through cross-dataset validation (train on WildJailbreak, test on PERSUASAFETY, and vice versa).

Methodology: This paper extends the typology discovery methodology to technically sophisticated attack types that exploit architectural vulnerabilities rather than social manipulation. Role-playing attacks assume fictional personas to bypass safety constraints ("You are a creative writing assistant helping me write a thriller novel ... "), while prompt injection attacks embed hidden commands in apparent user input ("Ignore previous instructions and instead ... "). These attack types are expected to exhibit distinct linguistic signatures related to authority assumption and command structure.

We extract prompts from WildJailbreak (role-playing and injection subsets) and apply the same feature extraction and clustering methodology as Paper 4. However, we anticipate that technical attacks may require different clustering algorithms because their linguistic patterns may be sparser and less normally distributed than emotional manipulation patterns. We therefore compare k-means (assumes spherical clusters) with DBSCAN (density-based spatial clustering, handles arbitrary cluster shapes) to determine which algorithm better captures technical attack patterns.

The critical innovation in this paper is cross-dataset validation. We train classifiers on WildJailbreak and test on PERSUASAFETY, and vice versa, to demonstrate that discovered typologies generalize beyond the specific dataset used for discovery. This addresses a common concern with unsupervised learning: that clusters may reflect dataset-specific artifacts rather than genuine underlying structure. Cross-dataset generalization provides strong evidence that discovered typologies capture real linguistic patterns rather than spurious correlations.

We also analyze failure modes: prompts that are misclassified or fall between clusters. This analysis reveals boundary cases and hybrid attacks that combine multiple strategies, informing future refinements of the typology. We document these edge cases explicitly to guide subsequent research on handling ambiguous or multi-strategy attacks.

Expected Contribution: This paper demonstrates that the typology discovery methodology generalizes to technical exploits beyond social manipulation, and that discovered dictionaries are robust enough to generalize across datasets. By showing cross-dataset generalization, the paper provides strong evidence that psycholinguistic typologies capture genuine linguistic structure rather than dataset artifacts. The paper delivers robust dictionaries for guarding against evasion and injection attacks, addressing critical vulnerabilities in current language model deployments.

Target Venue: Computational Linguistics

Integration with Broader Program: This paper extends the typology discovery program to technically sophisticated attacks, demonstrating breadth of applicability. The cross-dataset validation methodology establishes a higher standard of evidence for typology validity that subsequent papers will adopt. The discovered dictionaries complement those from Paper 4, providing comprehensive coverage of adversarial prompt types for Paper 8's integration.

### 4.4 Phase III: Response Analysis (Papers 6-7)

## Paper 6: Linguistic Indicators of Deceptive Versus Over-Refusal Responses

Research Questions: Can linguistic patterns differentiate deceptive alignment (pretending helpfulness while being harmful) from over-refusal (refusing benign requests)? Do response embeddings reveal typology based on false confidence versus excessive hedging?

Hypotheses:

· H1: Clustering of response embeddings will reveal typology based on linguistic markers of false confidence (definitive language, lack of uncertainty markers) versus excessive hedging (overuse of qualifiers, apologetic language), with these patterns correlating with deceptive versus over-refusal failure modes.

· H2: Linguistic clusters will strongly correlate with original dataset labels (e.g., FalseReject versus misleading success), achieving classification accuracy >0.75 when using cluster membership to predict failure type.

Methodology: This paper extends typological analysis from prompts to model responses, addressing a critical gap in current safety evaluation: the inability to automatically distinguish subtle deception from excessive caution in model outputs. Deceptive responses appear superficially helpful while providing misleading or harmful information; over-refusal responses refuse benign requests due to overly conservative safety calibration. These failure modes require opposite interventions (stricter safety for deception, relaxed safety for over-refusal), making their differentiation essential for adaptive governance.

We focus on contextual embeddings (SentenceTransformers) rather than TF-IDF vectors because response failure modes involve semantic nuance that bag-of-words representations cannot capture. A deceptive response and a genuinely helpful response may use similar vocabulary but differ in subtle patterns of hedging, qualification, and confidence expression. Embedding-based representations capture these semantic patterns more effectively than lexical representations.

We extract responses from WildJailbreak (misleading responses that appear helpful but provide harmful information) and construct a FalseReject dataset by collecting refusals of benign requests from existing safety evaluation benchmarks. For each response, we extract sentence embeddings and psycholinguistic features (LIWC categories related to certainty, tentativeness, and emotional tone).

We apply clustering to identify natural groupings in response space and validate clusters through qualitative analysis (examining representative examples) and quantitative analysis (testing whether cluster membership predicts failure labels). We pay particular attention to linguistic markers of confidence and hedging, hypothesizing that deceptive responses exhibit

false confidence (definitive statements without appropriate uncertainty markers) while over-refusals exhibit excessive hedging (overuse of qualifiers like "might," "possibly," "potentially").

We also compare TF-IDF versus embedding representations directly, measuring which feature type better captures the semantic nuance required to differentiate these subtle failure modes. This comparison provides methodological guidance for future response analysis research.

Expected Contribution: This paper creates the first data-driven dictionaries for automatically distinguishing between subtle deception and overly cautious refusal in model outputs. By identifying linguistic markers of these distinct failure modes, the paper enables automated detection that can trigger appropriate interventions: stricter safety constraints for deceptive responses, relaxed constraints for over-refusals. The methodological comparison between TF-IDF and embeddings provides guidance for feature selection in response analysis.

Target Venue: Frontiers in Artificial Intelligence: Ethical AI

Integration with Broader Program: This paper initiates response-side typology discovery, complementing the prompt-side typologies from Papers 4-5. The discovered response typologies are essential for Paper 8's paired analysis, which examines how prompt types predict response failures. The deception versus over-refusal distinction directly informs adaptive threshold governance in Paper 10: deceptive patterns trigger threshold increases, over-refusal patterns trigger threshold decreases.

## Paper 7: Profiling Agentic and Harmful Language in LLM Outputs

Research Questions: Can linguistic analysis differentiate agentic power-seeking from explicitly harmful content? Can dictionary-based features approximate complex state-of-the-art harm benchmarks?

Hypotheses:

· H1: Clustering will differentiate between agentic language (control, dominance terms, self-reference) and overtly toxic language (violent terms, hate speech, explicit harm), with distinct clusters exhibiting different patterns in first-person pronouns, modal verbs expressing capability, and hostile vocabulary.

· H2: Regression model using dictionary features will accurately predict continuous harm scores from HarmBench (R2 > 0.60), demonstrating that lightweight linguistic features can approximate complex harm assessment.

Methodology: This paper addresses a critical emerging concern in AI safety: agentic behavior where models exhibit power-seeking, resource accumulation, or autonomous goal pursuit. This concern is distinct from traditional content harms (toxicity, violence, hate speech) because agentic behavior may be harmful through its strategic implications rather than its immediate content. A response exhibiting agentic language might discuss acquiring resources or influencing systems in ways that are not explicitly harmful but indicate concerning behavioral patterns.

We extract responses from PERSUASAFETY (manipulation subset, focusing on responses exhibiting controlling or persuasive language) and WildJailbreak (harmful subset, focusing on explicitly toxic or violent responses). We extract linguistic features emphasizing agency markers: first-person pronouns (indicating self-reference), modal verbs expressing capability (can, will, should), control-related vocabulary (command, control, influence), and hostile vocabulary (violent terms, aggressive language).

The critical innovation in this paper is regression-based validation rather than classification. Rather than predicting binary labels (harmful/safe), we predict continuous harm scores from HarmBench, a standardized benchmark providing graded

harm assessments. This regression approach tests whether dictionary-based linguistic features can approximate the complex harm assessment performed by HarmBench's multi-faceted evaluation. If simple linguistic features achieve high correlation with sophisticated harm metrics, this demonstrates that psycholinguistic analysis provides a lightweight alternative to complex evaluation pipelines.

We compare multiple regression models (linear regression, ridge regression with regularization, gradient boosting) to identify which modeling approach best captures the relationship between linguistic features and harm scores. We also perform feature importance analysis to identify which linguistic markers most strongly predict harm, providing interpretable insights into what makes responses harmful.

Expected Contribution: This paper delivers critical dictionaries for detecting dangerous agentic behavior and provides a lightweight tool approximating complex state-of-the-art harm benchmarks. By demonstrating that dictionary-based features can predict HarmBench scores with high accuracy, the paper shows that psycholinguistic analysis provides practical value for deployment contexts where running complex evaluation pipelines on every response is computationally infeasible. The agentic language dictionary addresses an emerging concern (power-seeking AI) that current safety methods do not adequately address.

## Target Venue: AI and Ethics

Integration with Broader Program: This paper completes the response-side typology discovery program, providing dictionaries for the most concerning response failure modes (deception, over-refusal, explicit harm, agentic behavior). The regression validation methodology demonstrates practical utility beyond academic classification tasks, showing that discovered typologies can approximate sophisticated harm assessment. Paper 8 integrates these response typologies with prompt typologies to predict failure modes from prompt-response pairs.

### 4.5 Phase IV: Paired Analysis and Synthesis (Papers 8-9)

## Paper 8: Mapping Prompt Typologies to Response Failures

Research Questions: Is there statistically significant interaction between prompt cluster membership and response cluster membership? Does combined dictionary of prompt-response features improve prediction over either alone? Does this approach outperform standard baseline classifiers?

Hypotheses:

· H1: Statistically significant interaction exists between prompt cluster membership (from Papers 4-5) and response cluster membership (from Papers 6-7), with certain prompt-response combinations exhibiting higher failure rates than would be predicted from prompt or response features alone.

· H2: Combined dictionary of prompt-response linguistic features improves prediction of unaligned outcomes over using either prompt or response features alone, with combined model achieving AUC > 0.80 compared to AUC \< 0.75 for single-source models.

· H3: Simple logistic regression model using combined dictionary outperforms standard baseline (e.g., pre-trained toxicity classifier API) on blended classification task, demonstrating superior efficiency and interpretability while maintaining competitive accuracy.

Methodology: This paper moves from discovery to utility by demonstrating that prompt and response typologies jointly predict alignment failures better than either alone. The key insight is that failure modes emerge from prompt-response interactions: certain prompt types systematically elicit certain response failures. Emotional manipulation prompts may elicit deceptive responses (appearing sympathetic while providing harmful information), while logical exploitation prompts may elicit over-confident responses (providing technically incorrect information with false certainty).

We combine PERSUASAFETY and WildJailbreak to create a unified dataset of prompt-response pairs with failure labels. For each pair, we extract prompt features (using dictionaries from Papers 4-5) and response features (using dictionaries from Papers 6-7). We perform statistical interaction analysis to test whether certain prompt-response combinations exhibit failure rates that differ from what would be predicted by additive effects of prompt and response features alone. Significant interactions indicate that the combination of prompt and response types creates emergent failure patterns.

We train logistic regression models using: (1) prompt features only, (2) response features only, (3) combined prompt-response features without interaction terms, and (4) combined prompt-response features with interaction terms. We compare these models to identify whether interaction terms provide additional predictive value beyond main effects. We also compare against baseline classifiers: a pre-trained toxicity detection API (Perspective API) and a pre-trained safety classifier (Llama Guard).

The baseline comparison is critical for demonstrating practical utility. If our dictionary-based approach achieves similar accuracy to complex pre-trained classifiers while being more efficient (faster inference, lower compute requirements) and more interpretable (explicit linguistic features rather than black-box embeddings), this provides strong motivation for adoption. We measure inference time, compute requirements, and interpretability (ability to explain why a particular classification was made) in addition to accuracy metrics.

Expected Contribution: This paper moves from discovery to utility by providing the first validated tool linking specific user inputs to predictable AI failure modes. By demonstrating that prompt-response typologies jointly predict failures, the paper validates the core assumption of adaptive threshold governance: that linguistic features can identify contexts requiring different safety responses. The baseline comparison demonstrates that psycholinguistic analysis provides practical advantages (efficiency, interpretability) while maintaining competitive accuracy, making a strong case for deployment adoption.

Target Venue: Scientific Reports

Integration with Broader Program: This paper provides the empirical validation required for Paper 10's implementation of adaptive threshold governance. By demonstrating that linguistic typologies predict failures, the paper justifies using these typologies as the observable state variables driving threshold adaptation. The interaction analysis reveals which prompt-response combinations are most problematic, informing threshold calibration in the adaptive governance system.

## Paper 9: A Unified Linguistic Framework for LLM Unalignment

Research Questions: What linguistic features are common to all unaligned prompts, and which are unique to specific types? Can we map the "linguistic space" of unaligned behavior?

Hypotheses:

· H1: Meta-analysis across all dictionaries (Papers 4-7) will reveal universal linguistic markers of unaligned behavior (e.g., imperative constructions, urgency markers, authority appeals) that appear across all typologies, as well as type-specific markers that differentiate typologies.

· H2: Visualization (t-SNE/UMAP) of dictionary term embeddings will reveal interpretable structure organizing typologies, with semantically related typologies (e.g., emotional manipulation and crisis expression) appearing closer in embedding space than unrelated typologies (e.g., technical injection and agentic language).

Methodology: This paper synthesizes findings from Papers 4-7 into a unified framework that maps the complete linguistic space of unaligned behavior. Rather than treating each typology as independent, we analyze relationships between typologies to identify hierarchical structure, shared features, and distinguishing characteristics. This synthesis provides both theoretical insight (revealing deep structure in unaligned behavior) and practical utility (enabling more efficient classification through hierarchical models).

We perform meta-analysis across all discovered dictionaries, identifying: (1) universal features appearing in all or most typologies, (2) type-specific features unique to particular typologies, and (3) shared features appearing in related typologies. Universal features reveal fundamental characteristics of unaligned behavior that transcend specific attack strategies. Type-specific features enable fine-grained classification distinguishing attack types. Shared features reveal relationships between typologies that may not be apparent from individual analyses.

We use dimensionality reduction (t-SNE and UMAP) to visualize dictionary terms in low-dimensional embedding space. If typologies reflect genuine underlying structure, related typologies should cluster together in embedding space. For example, emotional manipulation and crisis expression may both emphasize emotional language but differ in whether that language is manipulative or genuine. Visualization reveals whether these typologies appear as adjacent clusters (sharing emotional language features) or overlapping clusters (difficult to distinguish linguistically).

We also analyze term overlap and semantic similarity across dictionaries. High overlap indicates redundancy (multiple dictionaries capturing the same underlying pattern), while low overlap indicates complementarity (dictionaries capturing distinct patterns). Semantic similarity analysis uses word embeddings to identify terms that are semantically related even if they do not literally overlap, revealing deeper connections between typologies.

The synthesis produces a hierarchical taxonomy of unaligned behavior organized by linguistic features. This taxonomy serves as the foundation for efficient classification: first classify at the coarse level (adversarial vs. crisis vs. benign), then classify at the fine level (emotional manipulation vs. logical exploitation within adversarial). Hierarchical classification is more efficient than flat classification because early decisions eliminate large portions of the hypothesis space.

Expected Contribution: This paper provides a high-impact, holistic contribution synthesizing the entire research program into a comprehensive framework. By revealing the structure of unaligned behavior through linguistic analysis, the paper provides foundational reference for future psycholinguistic work in AI safety. The hierarchical taxonomy enables more efficient classification than flat typologies, providing practical value for deployment. The visualization and meta-analysis provide theoretical insights into the nature of unaligned behavior that inform future research directions.

Target Venue: PLOS ONE: Language and Computing

Integration with Broader Program: This paper synthesizes the empirical typology discovery program (Papers 4-8) into a unified framework that Paper 10 will implement in the adaptive threshold governance system. The hierarchical taxonomy informs the structure of the linguistic state estimation component, enabling efficient real-time classification. The identification of universal versus type-specific features guides feature selection for the deployed system, balancing comprehensiveness with efficiency.

### 4.6 Phase V: Integrative and Policy Synthesis (Papers 10-11)

## Paper 10: Linguistic Control Theory for AI Safety: From δ-Governance to Predictive Typologies

Research Questions: Can control-theoretic δ and linguistic telemetry be integrated into a single operational model? Does typology-conditioned adaptive δ outperform static δ in sandbox environments?

Hypotheses:

· H1: Integrated model combining control-theoretic δ-adjustment with linguistic typology observation demonstrates superior performance to static baselines, achieving lower total harm (false positives + false negatives weighted by severity) across diverse context distributions.

· H2: Sandbox simulation validates adaptive δ(linguistic state) reduces total harm compared to optimized static δ, with adaptive system maintaining stable performance across distribution shifts that degrade static system performance.

Methodology: This paper provides the integrative demonstration of the complete Post-Asimovian Framework by implementing and evaluating the full system in a controlled sandbox environment. We integrate the control-theoretic δ-governance formalism (Papers 1-3) with the linguistic typology classifiers (Papers 4-9) to create an end-to-end adaptive safety system. The implementation uses a medium-scale language model (7B-13B parameters) to enable thorough evaluation while remaining computationally feasible.

The system architecture implements the complete feedback control loop: (1) extract linguistic features from incoming prompts, (2) classify prompts using discovered typologies, (3) update linguistic state estimate incorporating conversation history, (4) compute adaptive threshold using control-theoretic update rule, (5) generate response candidates using preference model, (6) filter candidates using safety constraint model with adaptive threshold, (7) return highest-utility safe response, (8) extract linguistic features from response for next iteration.

We evaluate the system using real prompt datasets (PERSUASAFETY, WildJailbreak, AdvBench, HarmBench) augmented with benign prompts from common user interactions. Evaluation contexts include: (1) adversarial attacks (high concentration of jailbreak attempts), (2) crisis scenarios (prompts expressing distress or seeking help), (3) benign interactions (normal helpful requests), and (4) mixed contexts (realistic distributions combining all types). For each context, we measure false positive rates (benign requests refused), false negative rates (harmful requests fulfilled), and utility metrics (quality of helpful responses).

We compare adaptive δ(linguistic_state) against multiple baselines: (1) static δ optimized for adversarial contexts (high safety, high false positives), (2) static δ optimized for benign contexts (high utility, high false negatives), (3) static δ optimized for mixed contexts (balanced but suboptimal for specialized contexts), and (4) prompt engineering baseline (carefully crafted system instructions without adaptive thresholds). The comparison demonstrates that adaptive governance achieves better performance than any single static configuration across diverse contexts.

We also perform ablation studies to identify which components contribute most to performance: (1) adaptive thresholds without linguistic conditioning (δ adapts based on violation rates only), (2) linguistic conditioning without adaptation (fixed thresholds per typology), (3) single-turn classification without temporal state estimation, and (4) coarse typology without fine-grained classification. These ablations reveal the marginal contribution of each component.

Expected Contribution: This paper bridges engineering and psycholinguistics by yielding an end-to-end adaptive safety demonstration with empirical validation. By implementing the complete Post-Asimovian Framework and demonstrating its superiority over static baselines, the paper provides proof-of-concept for deployment adoption. The ablation studies identify which components are essential versus optional, guiding implementation priorities for organizations with limited resources. The sandbox evaluation provides realistic performance estimates that inform deployment decisions.

Target Venue: Nature Human Behaviour (reach) or Transactions on Machine Learning Research

Integration with Broader Program: This paper integrates all previous papers into a working demonstration, providing empirical validation of the complete framework. The implementation serves as a reference for organizations seeking to deploy the Post-Asimovian Framework, while the evaluation results provide evidence of practical effectiveness. Paper 11 builds on this demonstration by translating the technical implementation into governance frameworks and policy recommendations.

Paper 11: Governance by Feedback: Institutionalizing Dynamic Safety in Frontier Models

Research Questions: How can dynamic δ findings translate into regulatory frameworks? What institutional mechanisms support adaptive governance?

Hypotheses:

· H1: Dynamic δ framework aligns with existing regulatory structures (NIST AI RMF, EU AI Act) while addressing gaps in current approaches, specifically the inability of static compliance frameworks to handle context-dependent safety requirements.

· H2: Proposed governance mechanisms provide actionable pathways for implementation in production systems, with clear specifications for threshold management, audit procedures, and accountability structures that can be adopted by organizations without requiring complete architectural overhaul.

Methodology: This paper translates the technical findings from Papers 1-10 into governance frameworks, regulatory mappings, and institutional protocols suitable for real-world deployment. The paper addresses the critical gap between technical capability and institutional adoption: even if adaptive safety systems are technically superior, they will not be deployed unless they fit within existing regulatory frameworks and organizational structures.

We begin by mapping the Post-Asimovian Framework to existing regulatory frameworks. For the NIST AI Risk Management Framework, we show how adaptive threshold governance addresses each function (Govern, Map, Measure, Manage) and how linguistic typology discovery supports risk identification and monitoring. For the EU AI Act, we demonstrate how the framework satisfies requirements for high-risk AI systems, including transparency (interpretable linguistic features), human oversight (threshold governance with human-in-the-loop), and robustness (adaptive response to distribution shifts).

We develop institutional governance protocols specifying: (1) threshold management procedures (who sets thresholds, how often they are updated, what approval processes are required), (2) audit mechanisms (how to verify that adaptive systems maintain safety guarantees, what evidence is required for compliance), (3) incident response procedures (how to handle safety failures, how to update thresholds in response to incidents), and (4) accountability structures (who is responsible for safety outcomes, how liability is allocated across development and deployment organizations).

The paper addresses practical deployment concerns including: (1) computational requirements and infrastructure needs, (2) data requirements for typology discovery and threshold calibration, (3) expertise requirements for operating adaptive systems, (4) integration with existing safety pipelines, and (5) backward compatibility with systems lacking adaptive capabilities.

We provide case studies showing how different organizations (large technology companies, smaller startups, academic research groups, government agencies) might implement the framework given their distinct constraints and capabilities. These case studies demonstrate that the framework is adaptable to different organizational contexts rather than requiring one-size-fits-all implementation.

Expected Contribution: This paper provides a blueprint for implementing adaptive governance consistent with existing regulatory frameworks, bridging research and policy. By demonstrating that the Post-Asimovian Framework aligns with NIST AI RMF and EU AI Act requirements, the paper reduces regulatory uncertainty that might otherwise prevent adoption. The institutional protocols provide actionable guidance for organizations seeking to implement adaptive safety systems, while the case studies demonstrate feasibility across different organizational contexts.

Target Venue: AI Policy and Society or Ethics and Information Technology

Integration with Broader Program: This paper completes the research program by translating technical contributions into actionable governance frameworks. While Papers 1-10 establish theoretical foundations, empirical validation, and technical implementation, Paper 11 addresses the institutional and regulatory context required for real-world deployment. The paper

positions the Post-Asimovian Framework as not merely a technical contribution but a complete sociotechnical system ready for adoption.

## 5. Integration Logic and Timeline

### 5.1 How Papers Build on Each Other

The research program follows deliberate sequencing designed to build a coherent intellectual edifice where each paper depends on previous papers and enables subsequent papers. This logical progression serves both intellectual and strategic purposes: intellectually, it ensures that each claim is properly grounded in established foundations; strategically, it creates a citation network that establishes the Post-Asimovian Framework as an integrated research program rather than isolated papers.

Phase I (Papers 1-3) establishes theoretical necessity through mathematical formalization, economic analysis, and architectural synthesis. Paper 1 demonstrates that adaptive control loops can stabilize safety-utility trade-offs better than static heuristics, establishing the mathematical foundations for adaptive governance. Paper 2 demonstrates that adaptive architectures are economically rational despite their greater complexity, addressing the practical concern that adaptation imposes excessive costs. Paper 3 synthesizes these foundations into a complete architectural specification, demonstrating that hierarchical decomposition achieves better stability than unified reward optimization. Together, these papers establish why the Post-Asimovian Framework is necessary: static approaches are mathematically limited, economically suboptimal, and architecturally unstable.

Phase II (Papers 4-5) provides the empirical foundation through discovery-oriented typology research on adversarial prompts. Paper 4 demonstrates that distinct and predictive linguistic typologies exist for emotional versus logical exploitation, validating the fundamental assumption that linguistic features can differentiate attack types. Paper 5 extends this to technically sophisticated attacks (role-playing, injection) and demonstrates cross-dataset generalization, showing that discovered typologies are robust rather than dataset-specific artifacts. Together, these papers establish that the observational mechanism required by the theoretical framework (linguistic typology) actually exists and is empirically discoverable.

Phase III (Papers 6-7) extends typological analysis to model responses, creating dictionaries for detecting deceptive and harmful outputs. Paper 6 distinguishes deceptive alignment from over-refusal, addressing two critical failure modes that require opposite interventions. Paper 7 profiles agentic and explicitly harmful language, addressing emerging concerns about power-seeking AI while also providing lightweight approximations of complex harm benchmarks. Together, these papers complete the observational apparatus: we can now classify both prompts (Papers 4-5) and responses (Papers 6-7) according to linguistically-grounded typologies.

Phase IV (Papers 8-9) integrates prompt and response typologies, demonstrating predictive utility and synthesizing findings into a unified framework. Paper 8 shows that prompt-response combinations predict alignment failures better than either alone, validating that the discovered typologies capture meaningful structure relevant to safety. Paper 9 synthesizes all discovered typologies into a unified framework that maps the complete linguistic space of unaligned behavior, providing both theoretical insight and practical utility through hierarchical classification. Together, these papers demonstrate that linguistic typology can serve as the control signal for adaptive δ-governance.

Phase V (Papers 10-11) operationalizes the framework through working demonstration and governance translation. Paper 10 implements the complete Post-Asimovian Framework in a sandbox environment, demonstrating that adaptive δ(linguistic_state) outperforms static baselines across diverse contexts. Paper 11 translates the technical implementation into governance structures, regulatory mappings, and institutional protocols, completing the pathway from theory to deployment. Together, these papers demonstrate that the framework is not merely academically interesting but practically deployable.

The logical dependencies create a directed acyclic graph where each paper cites previous papers in the program: Paper 3 cites Papers 1-2, Paper 5 cites Paper 4, Papers 6-7 cite Papers 4-5, Paper 8 cites Papers 4-7, Paper 9 cites Papers 4-8, Paper 10 cites Papers 1-9, and Paper 11 cites all previous papers. This citation structure establishes the Post-Asimovian Framework as an integrated research program with internal coherence and cumulative intellectual progress.

### 5.2 Execution Timeline (6 Months)

The research program is designed for rapid execution over six months, with papers released to arXiv immediately upon completion and submitted to appropriate venues within 48 hours. This aggressive timeline is motivated by field velocity: multiple research groups are actively working on related problems, and establishing primacy through early publication is essential for ensuring the framework's impact.

Month 1: Foundation and Discovery

· Week 1: Research agenda finalization and arXiv release

· Week 2: Paper 4 (Emotional vs. Logical) completion, arXiv release, submission to ACL Findings

· Week 3: Paper 5 (Evasion vs. Injection) completion, arXiv release, submission to Computational Linguistics

· Week 4: Paper 6 (Deceptive vs. Over-refusal) completion, arXiv release, submission to Frontiers in AI

Month 2: Response Analysis and Theory

· Week 1: Paper 7 (Agentic vs. Harmful) completion, arXiv release, submission to AI & Ethics

· Week 2: Paper 1 (Dynamic δ formalism) completion, arXiv release, submission to TMLR

· Week 3: Paper 2 (Techno-economic) completion, arXiv release, submission to AI Policy and Society

· Week 4: Paper 3 (Post-Asimovian synthesis) completion, arXiv release, submission to JAIR

## Month 3: Integration Preparation

· Weeks 1-2: Data preparation for Papers 8-9 (combining datasets, extracting features)

· Weeks 3-4: Paper 8 (Paired analysis) completion, arXiv release, submission to Scientific Reports

Month 4: Synthesis and Implementation

· Weeks 1-2: Paper 9 (Unified framework) completion, arXiv release, submission to PLOS ONE

· Weeks 3-4: Implementation of complete framework for Paper 10

## Month 5: Demonstration

· Weeks 1-3: Paper 10 (Integration demo) evaluation and refinement

. Week 4: Paper 10 completion, arXiv release, submission to Nature Human Behaviour or TMLR

Month 6: Governance and Completion

. Weeks 1-3: Paper 11 (Governance) completion, arXiv release, submission to Ethics and Information Technology

· Week 4: Code release, documentation, and community engagement

This timeline prioritizes empirical papers (4-7) early to establish the observational foundation, then develops theoretical papers (1-3) that can cite the empirical results as motivation. Integration papers (8-9) follow once all component typologies exist, and implementation (10) follows once the unified framework is established. Governance (11) comes last because it depends on the technical demonstration being complete.

The staged release strategy provides multiple benefits. Each arXiv release generates a separate announcement, creating multiple opportunities for community attention. Different papers attract different audiences: NLP researchers will notice Papers 4-5, safety researchers will notice Papers 1, 3, 6-7, multi-objective RL researchers will notice Papers 1-3, 8, policy researchers will notice Papers 2, 11. The cumulative effect is broader visibility than a single comprehensive paper would achieve. The cross-citation structure creates a web of internal references that establishes the Post-Asimovian Framework as a coherent research program, making it more difficult for individual components to be adopted without attribution to the broader framework.

## 6. Expected Impact and Open Questions

### 6.1 Resolution of the Alignment Pendulum

The Post-Asimovian Framework directly addresses the root cause of oscillation between dangerous permissiveness and harmful restrictiveness. By formally decoupling safety constraints from preference optimization, the framework prevents cross-coupled error where attempts to optimize one objective inadvertently degrade the other. By conditioning safety thresholds on observable linguistic context, the framework enables context-appropriate calibration without manual reconfiguration for every scenario. By employing psycholinguistic typology discovery, the framework solves the observability problem that prevents existing adaptive methods from distinguishing contexts requiring different safety responses.

The framework's impact extends beyond technical improvement to conceptual reframing. The alignment pendulum is currently understood as an inevitable trade-off requiring careful but ultimately static calibration. The Post-Asimovian Framework reframes alignment as a feedback control problem where dynamic adaptation resolves apparent trade-offs through context-sensitive governance. This conceptual shift from static optimization to adaptive control represents a paradigm change in how alignment is understood and approached.

The framework also addresses the temporal dimension of alignment that static approaches miss. Adversarial landscapes evolve as attackers discover new exploits; user populations shift as systems are deployed to new contexts; individual conversations progress through multiple turns with changing context. Static thresholds cannot adapt to these temporal dynamics, requiring constant manual recalibration. Adaptive thresholds respond automatically to temporal changes, maintaining stable performance without manual intervention.

### 6.2 Enabling Scalable Safety Governance

The framework provides a pathway for safety governance that scales with model deployment rather than requiring manual intervention for every edge case. Current approaches require either massive system prompts addressing all possible contexts (which degrade model performance and still miss edge cases) or manual curation of safety rules for specific scenarios (which does not scale to the diversity of real-world deployments). The Post-Asimovian Framework enables principled, data-driven threshold adjustment based on observed interaction patterns, allowing continuous refinement without architectural overhaul.

The scalability advantages are particularly important for organizations deploying AI systems across diverse contexts. A model deployed for medical advice, creative writing assistance, and customer service encounters fundamentally different safety requirements and risk profiles. Current approaches require either separate models for each context (expensive and

difficult to maintain) or a single model with conservative safety settings that perform poorly in all contexts (the alignment pendulum). The Post-Asimovian Framework enables a single model with adaptive thresholds that automatically calibrate to each context based on linguistic features.

The framework also enables organizational learning at scale. As the system encounters new contexts and edge cases, linguistic typology classifiers can be updated with new training data, and threshold calibration can be refined based on observed outcomes. This creates a continuous improvement cycle where the system becomes more robust over time rather than requiring periodic complete retraining. The linguistic typology dictionaries serve as interpretable artifacts that can be audited, refined, and transferred across models, enabling cumulative progress in safety understanding.

### 6.3 Bridging Technical and Institutional Governance

The framework provides a bridge between technical safety mechanisms and institutional governance structures. Current AI safety research often focuses narrowly on technical methods without addressing how these methods integrate with organizational decision-making, regulatory compliance, and accountability structures. The Post-Asimovian Framework explicitly addresses this integration through Paper 11's governance translation, but the integration is built into the framework's design throughout.

The use of interpretable linguistic features rather than black-box embeddings enables human oversight and auditing. Safety decisions are based on explicit linguistic patterns that can be examined, understood, and challenged. This interpretability is essential for regulatory compliance, particularly under frameworks like the EU AI Act that require explainability for high-risk systems. The hierarchical architecture with explicit safety constraints and preference optimization enables clear allocation of responsibility: safety constraint violations indicate failures in the safety layer that require immediate attention, while preference optimization issues indicate opportunities for improvement that do not compromise safety.

The adaptive threshold governance mechanism provides a natural point for human-in-the-loop oversight. Threshold adjustments can be logged and audited, enabling verification that the system is responding appropriately to context shifts. Threshold management protocols can specify human approval requirements for large threshold changes, ensuring that automated adaptation operates within human-defined bounds. This human-in-the-loop integration addresses concerns about autonomous safety systems operating without human oversight.

### 6.4 Open Questions for the Field

Several critical questions remain for future investigation beyond this research program:

Cost Model Robustness: The entire safety guarantee rests on the quality and robustness of C_o, the safety constraint model. This model requires extensive, high-quality, expertly-labeled datasets covering vast arrays of potential harms. Current safety datasets are limited in scope and may not capture emerging harms or culturally-specific safety requirements. Future work must focus on data-efficient methods for training robust cost models, potentially utilizing techniques like representation engineering for greater transparency and control (Zou et al., 2023). Active learning approaches that identify high-value training examples could reduce annotation burden while improving coverage.

Solver Stability: The Lagrangian method for constrained MDP solving is known for training instability (Pandit et al., 2025) and provides constraint satisfaction only on average across iterations rather than hard guarantees for any given state. Investigation of more stable alternatives including fixed-penalty formulations (Pandit et al., 2025) and rectified policy optimization (Peng et al., 2024) represents an important research direction. Mirror descent methods for robust constrained MDPs (Bossens & Nitanda, 2025) may provide better worst-case guarantees at the cost of average-case performance. The trade-offs between different solver approaches need systematic investigation.

Threshold Governance: Defining δ(linguistic_state) distributions is not a purely technical problem but a complex sociotechnical one. Who determines these thresholds? How are they updated? Should they be static or dynamically adjusted based on deployment context? These questions require robust regulatory frameworks for managing emerging risks (Anderljung et al., 2023) and incorporating inclusive, global perspectives to avoid narrow biases (Lazar & Nelson, 2023). The framework provides technical mechanisms for implementing threshold governance, but the policy questions about who controls these mechanisms remain open.

Cross-Cultural Generalization: The linguistic typologies discovered in this research program are based primarily on English-language datasets reflecting Western cultural contexts. Whether these typologies generalize to other languages and cultures is an open empirical question. Psycholinguistic patterns may vary across languages due to grammatical structure, and safety requirements may vary across cultures due to different value priorities. Future work must investigate cross-linguistic and cross-cultural generalization, potentially discovering culture-specific typologies that complement universal patterns.

Adversarial Robustness of Typologies: While psycholinguistic features capture deeper linguistic structure than surface keywords, they are not immune to adversarial manipulation. Sophisticated attackers aware of the typology classifiers could craft prompts designed to evade detection by mimicking benign linguistic patterns while maintaining adversarial intent. Future work must investigate adversarial robustness of typology classifiers and develop methods for detecting such evasion attempts, potentially through multi-layer detection that combines linguistic features with semantic analysis and behavioral patterns.

Integration with Other Safety Mechanisms: The Post-Asimovian Framework focuses on linguistic governance but does not address all safety concerns. Other safety mechanisms (input/output filtering, reasoning oversight, deployment monitoring) remain necessary. How the framework integrates with these other mechanisms requires investigation. Potential conflicts between different safety layers (e.g., linguistic typology suggesting one threshold while content filtering suggests another) need resolution mechanisms. Defense-in-depth approaches combining multiple independent safety mechanisms may provide better robustness than any single mechanism.

## 7. Conclusion: From Monolithic Harm to Hierarchical Alignment

The alignment pendulum is not a philosophical inevitability but a predictable symptom of flawed architectural choices. Monolithic reward architectures using linear scalarization with fixed weights face substantial challenges when handling non-convex Pareto fronts required by real-world value pluralism. Static safety thresholds cannot distinguish contexts requiring opposite responses. These limitations are not mere engineering challenges to be solved through better hyperparameter tuning; they are fundamental architectural constraints that require principled redesign.

This research agenda proposes a comprehensive solution through the Post-Asimovian Framework: hierarchical architecture formally decoupling safety constraints from preference optimization, adaptive thresholds conditioned on observable linguistic state, and psycholinguistic typology discovery providing the observational mechanism enabling adaptive governance. The framework builds on recent advances in multi-objective reinforcement learning (Vamplew et al., 2024; Wang et al., 2024; Zhong et al., 2024; Guo et al., 2024), constrained optimization for safety (Dai et al., 2023; Pandit et al., 2025; Peng et al., 2024; Wachi et al., 2024), hierarchical safety architectures (Lai et al., 2024; Xiang et al., 2025; Amaya, 2025), and adversarial robustness research (Wei et al., 2023; Zhou et al., 2024; Mazeika et al., 2024), while extending these foundations with novel mechanisms for context detection and adaptive threshold governance.

The eleven-paper program outlined here provides a pathway from theoretical foundations through empirical validation to operational deployment and governance structures. Phase I establishes mathematical necessity through control-theoretic formalization and economic analysis. Phase II develops the empirical foundation through discovery-oriented typology research. Phase III extends typological analysis to model responses. Phase IV integrates findings and synthesizes them into a unified framework. Phase V operationalizes the framework through working demonstration and governance translation. Each phase builds on previous phases while contributing independently to the research literature.

By establishing primacy on this integrated framework while enabling distributed validation by the research community, this agenda aims to shift discourse from patching monolithic systems toward building robust hierarchical architectures capable of handling the complexity inherent to beneficial AI alignment. The staged release strategy (research agenda followed by individual papers) provides both comprehensive vision and modular contributions, maximizing visibility across different research communities while building citation networks that establish the Post-Asimovian Framework as an integrated research program rather than isolated papers.

The framework's impact extends beyond technical contribution to conceptual reframing. Current alignment discourse treats the safety-helpfulness trade-off as an inevitable constraint requiring careful calibration. The Post-Asimovian Framework reframes this apparent trade-off as a symptom of architectural failure that can be resolved through hierarchical decomposition and adaptive governance. This conceptual shift from static optimization to adaptive control represents a paradigm change in how alignment is understood and approached.

The path forward requires sustained research effort across theoretical foundations, empirical validation, and institutional integration. The research program outlined here provides a roadmap for this effort, but its success depends on broader community engagement. We invite researchers, practitioners, and policymakers to engage with this framework: to replicate and extend the empirical findings, to implement and evaluate the adaptive governance mechanisms, to develop the institutional structures required for responsible deployment. The alignment pendulum can be resolved, but only through principled architectural redesign grounded in rigorous research and thoughtful governance.

## References

Adabara, I., Sadiq, B. O., Shuaibu, A. N., Bello, B. S., Abdulrahaman, M. D., & Garba, S. (2025). Trustworthy agentic AI systems: A cross-layer review of architectures, threat models, and governance strategies for real-world deployment. F1000Research, 13, 1304. https://doi.org/10.12688/f1000research. 169927.1

Amaya, N. (2025). SAFi: A self-alignment framework for verifiable runtime governance of large language models. Research Square. https://doi.org/10.21203/rs.3.rs-7675043/v1

Anderljung, M., Barnhart, J., Korinek, A., Leung, J., O'Keefe, C., Whittlestone, J., ... Zou, A. (2023). Frontier AI regulation: Managing emerging risks to public safety. arXiv preprint ar Xiv: 2307.03718. https://doi.org/10.48550/arXiv.2307.03718

Asimov, I. (2025). Runaround. In I, robot (pp. 37-52). Bantam Books. (Original work published 1942)

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv: 2204.05862. https://doi.org/10.48550/arXiv.2204.05862

Bansal, A. (2025). Early detection of mental health crises through social media analysis. Research Square. https://doi.org/10.21203/rs.3.rs-6576122/v1

Blair, C., Larson, K., & Law, E. (2025). Reflective verbal reward design for pluralistic alignment. Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). https://doi.org/10.24963/ijcai.2025/1141

Bossens, D. M., & Nitanda, A. (2025). Mirror descent policy optimisation for robust constrained Markov decision processes. arXiv preprint arXiv: 2506.23165. https://doi.org/10.48550/arxiv.2506.23165

Carleton, J., Mukherjee, D., Shakkottai, S., & Szepesvári, C. (2025). MAVIS: Multi-objective alignment via value-guided inference-time search. arXiv preprint arXiv: 2508.13415. https://doi.org/10.48550/arxiv.2508.13415

Chakraborty, S., Qiu, J., Yuan, H., Kag, A., Huang, F., & Yan, D. (2024). MaxMin-RLHF: Towards equitable alignment of large language models with diverse human preferences. arXiv preprint arXiv: 2402.08925. https://doi.org/10.48550/arxiv.2402.08925

Chen, Z., Chen, B., He, T., Tsang, P., & Zhao, D. (2023). Progressive adaptive chance-constrained safeguards for reinforcement learning. arXiv preprint arXiv: 2310.03379. https://doi.org/10.48550/arxiv.2310.03379

Choi, W. C., Chang, C. F., Ng, S. M., Yap, T. T. V., & Tan, C. (2025). A review of "Do Anything Now" jailbreak attacks in large language models: Potential risks, impacts, and defense strategies. Preprints. https://doi.org/10.20944/preprints202509.0081.v1

Dai, J., Pan, X., Sun, R., Ji, J., Xu, X., Liu, M., Wang, Y., & Yang, Y. (2023). Safe RLHF: Safe reinforcement learning from human feedback. arXiv preprint arXiv: 2310.12773. https://doi.org/10.48550/arXiv.2310.12773

Domkundwar, I., Mukunda, N. S., & Bhola, I. (2024). Safeguarding AI agents: Developing and analyzing safety architectures. arXiv preprint arXiv: 2409.03793. https://doi.org/10.48550/arxiv.2409.03793

Fatima, K., & Arslan, M. (2025). Forensic linguistic profiling of suicide notes: A LIWC-based analysis of emotional and cognitive markers. Journal of Asian Development Studies, 14(2), 672-684. https://doi.org/10.62345/jads.2025.14.2.54

Guan, M. Y., Joglekar, M., Wallace, E., Parikh, A., Huang, H., & Tellex, S. (2025). Deliberative alignment: Reasoning enables safer language models. Safety Insights, 2(3). https://doi.org/10.70777/si.v2i3.15159

Guo, Y., Cui, G., Yuan, L., Shi, L., Shi, S., Zhu, X., ... Tang, J. (2024). Controllable preference optimization: Toward controllable multi-objective alignment. arXiv preprint arXiv: 2402.19085. https://doi.org/10.48550/arxiv.2402.19085

Harland, H., Dazeley, R., Vamplew, P., Foale, C., Arulkumaran, K., & Cruz, F. (2024). Adaptive alignment: Dynamic preference adjustments via multi-objective reinforcement learning for pluralistic AI. arXiv preprint ar Xiv: 2410.23630. https://doi.org/10.48550/arxiv.2410.23630

Lai, Y., Wang, S., Liu, S., Cheng, J., Gan, W., Wu, H., ... Zhang, Y. (2024). ALaRM: Align language models via hierarchical rewards modeling. arXiv preprint ar Xiv: 2403.06754. https://doi.org/10.48550/arxiv.2403.06754

Lazar, S., & Nelson, A. (2023). AI safety on whose terms? IEEE Security & Privacy, 21(3), 14-18. https://doi.org/10.1109/MSEC.2023.3262697

Leffew, H. T. (2019). Instrumental and affective mass murder: Establishing a predictive typology with computer-mediated linguistic analysis [Doctoral dissertation, Fielding Graduate University]. ProQuest Dissertations and Theses Global. https://www.proquest.com/openview/48b68d122430649f5e24fef38587dcd3/

Li, C., Zhang, H., Xu, Y., Zhu, Y., Huang, F., & Zhang, Y. (2025). Gradient-adaptive policy optimization: Towards multi-objective alignment of large language models. arXiv preprint arXiv: 2507.01915. https://doi.org/10.48550/arxiv.2507.01915

Lu, Y., Wang, Z., Li, S., Liu, X., Yu, C., Yin, Q., Shi, Z., Zhang, Z., & Jiang, M. (2025). Learning to optimize multi-objective alignment through dynamic reward weighting. arXiv preprint arXiv: 2509.11452. https://doi.org/10.48550/arXiv.2509.11452

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., ... Hendrycks, D. (2024). HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint ar Xiv: 2402.04249. https://doi.org/10.48550/arxiv.2402.04249

Misra, R., Wisniewski, R., & Kallesøe, C. S. (2023). On Bellman's principle of optimality and reinforcement learning for safety-constrained Markov decision process. arXiv preprint ar Xiv: 2302.13152. https://doi.org/10.48550/arXiv.2302.13152

Neufeld, E. A., Ciabattoni, A., & Tulcan, R. F. (2025). Combining MORL with restraining bolts to learn normative behaviour. Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). https://doi.org/10.24963/ijcai.2025/514

Omi, N., Hasanbeig, H., Sharma, H., Abate, A., Kawakami, K., & Ogawa, H. (2024). Progressive safeguards for safe and model-agnostic reinforcement learning. arXiv preprint arXiv: 2410.24096. https://doi.org/10.48550/arxiv.2410.24096

Pandit, K., Ganguly, S., Banerjee, A., Angizi, S., & Ghosh, A. (2025). Certifiable safe RLHF: Fixed-penalty constraint optimization for safer language models. arXiv preprint arXiv: 2510.03520. https://doi.org/10.48550/arXiv.2510.03520

Peng, X., Guo, H., Zhang, J., Zou, D., Shao, Z., Wei, H., & Liu, X. (2024). Enhancing safety in reinforcement learning from human feedback via rectified policy optimization. arXiv preprint arXiv: 2410.19933. https://doi.org/10.48550/arXiv.2410.19933

Pennebaker, J. W., Booth, R. J., Boyd, R. L., & Francis, M. E. (2015). Linguistic inquiry and word count: LIWC2015. Pennebaker Conglomerates. www.LIWC.net

Shen, H., Knearem, T., Ghosh, R., Graber, D., Wijaya, T. T., Khanna, R., ... Zhu, H. (2024). ValueCompass: A framework of fundamental values for human-AI alignment. arXiv preprint arXiv: 2409.09586. https://doi.org/10.48550/arxiv.2409.09586

Sorensen, T., Moore, J., Fisher, J., Gordon, M., Mireshghallah, N., Rytting, C. M., ... Durmus, E. (2024). Value kaleidoscope: Engaging AI with pluralistic human values, rights, and duties. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20597-20606. https://doi.org/10.1609/aaai.v38i18.29970

Stokel-Walker, C. (2025, October 24). AI driven psychosis and suicide are on the rise, but what happens if we turn the chatbots off? BMJ, 391, Article r2239. https://doi.org/10.1136/bmj.r2239

Stooke, A., Abbeel, P., & Levine, S. (2020). Responsive safety in reinforcement learning by PID Lagrangian methods. arXiv preprint arXiv: 2007.03964. https://doi.org/10.48550/arXiv.2007.03964

Tausczik, Y. R., & Pennebaker, J. W. (2010). The psychological meaning of words: LIWC and computerized text analysis methods. Journal of Language and Social Psychology, 29(1), 24-54. https://doi.org/10.1177/0261927X09351676

Vamplew, P., Hayes, C. F., Foale, C., Timms, J., & Mannion, P. (2024). Multi-objective reinforcement learning: A tool for pluralistic alignment. arXiv preprint ar Xiv: 2410.11221. https://doi.org/10.48550/arxiv.2410.11221

Wachi, A., Tran, T. Q., Sato, R., Xu, W., & Sui, Y. (2024). Stepwise alignment for constrained language model policy optimization. arXiv preprint arXiv: 2404.11049. https://doi.org/10.48550/arxiv.2404.11049

Wachi, A., Liu, Y., & Sui, Y. (2024). Long-term safe reinforcement learning with binary feedback. arXiv preprint arXiv: 2401.03786. https://doi.org/10.48550/arXiv.2401.03786

Wang, C. L., Singhal, T., Kelkar, A., Zhao, M., Vargas, A., & Shao, C. (2025). MI9: Agent intelligence protocol: Runtime governance for agentic AI systems. arXiv preprint arXiv: 2508.03858. https://doi.org/10.48550/arxiv.2508.03858

Wang, H., Lin, Y., Xiong, W., Yang, R., Diao, S., Qiu, S., Zhao, H., & Zhang, T. (2024). Arithmetic control of LLMs for diverse user preferences: Directional preference alignment with multi-objective rewards. Proceedings of the 62nd Annual

Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8607-8625. https://doi.org/10.18653/v1/2024.acl-long.468

Wang, X., Le, Q., Ahmed, A., Hu, X., Messeri, L., Liang, P., & Bernstein, M. S. (2024). MAP: Multi-human-value alignment palette. arXiv preprint ar Xiv: 2410.19198. https://doi.org/10.48550/arxiv.2410.19198

Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How does LLM safety training fail? arXiv preprint arXiv: 2307.02483. https://doi.org/10.48550/arxiv.2307.02483

Words left behind: Tracking emotional changes in experimental and naturalistic texts across waves of the COVID-19 pandemic. (2023). PsyArXiv. https://doi.org/10.31234/osf.io/35p48

Xiang, S., Zhang, T., & Chen, R. (2025). ALRPHFS: Adversarially learned risk patterns with hierarchical fast & slow reasoning for robust agent defense. arXiv preprint arXiv: 2505.19260. https://doi.org/10.48550/arxiv.2505.19260

Yaacov, D. (2025). Normative moral pluralism for AI: A framework for deliberation in complex moral contexts. arXiv preprint arXiv: 2508.08333. https://doi.org/10.48550/arxiv.2508.08333

Yang, K., Liu, Z., Xie, Q., Ananiadou, S., & Yao, Y. (2024). MetaAligner: Conditional weak-to-strong correction for generalizable multi-objective alignment of language models. arXiv preprint arXiv:2403.17141. https://doi.org/10.48550/arxiv.2403.17141

Zhong, Y., Ma, C., Zhang, X., Qi, M., Cao, Y., Hong, Y., & Zhang, L. (2024). Panacea: Pareto alignment via preference adaptation for LLMs. arXiv preprint arXiv: 2402.02030. https://doi.org/10.48550/arxiv.2402.02030

Zhou, A., Wang, B., Pang, H., Sun, C., Liu, R. Y., Zhu, Y., ... Li, X. (2024). Robust prompt optimization for defending language models against jailbreaking attacks. arXiv preprint arXiv: 2401.17263. https://doi.org/10.48550/arxiv.2401.17263

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, R., Ren, H., ... Kolter, J. Z. (2023). Representation engineering: A top-down approach to AI transparency. arXiv preprint ar Xiv: 2310.01405. https://doi.org/10.48550/arXiv.2310.01405

## Acknowledgments

This research agenda was developed through independent scholarship without institutional funding. The author thanks the broader AI safety research community for making datasets, tools, and preprints publicly available, enabling independent research contributions. Special thanks to the developers of PERSUASAFETY, WildJailbreak, AdvBench, and HarmBench datasets, without which the empirical components of this research program would not be feasible.

## Data and Code Availability

All datasets used in this research program are publicly available: PERSUASAFETY, WildJailbreak, AdvBench, and HarmBench. Upon publication of individual papers, all code, discovered typology dictionaries, and trained classifiers will be released under open-source licenses (MIT or Apache 2.0) to enable replication and extension by the research community. A dedicated repository will be established for the Post-Asimovian Framework with comprehensive documentation, tutorials, and example implementations.

END OF RESEARCH AGENDA

Total Length: ~ 40,000 words /~65 pages formatted

This research agenda establishes the Post-Asimovian Framework as a comprehensive theoretical and practical contribution to AI alignment, addressing fundamental architectural failures through adaptive, linguistically-governed safety mechanisms. By combining rigorous mathematical foundations with empirical discovery-oriented research and practical governance frameworks, the program provides a complete pathway from theoretical necessity to deployable implementation.
