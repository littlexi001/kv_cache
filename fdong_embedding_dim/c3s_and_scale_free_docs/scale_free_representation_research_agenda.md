# Toward Scale-Free and Fully Utilized Representation Spaces in Large Language Models

## A Research Agenda From Singular Concentration to Distributed Semantic Capacity

## Abstract

The current working theory starts from a mechanistic observation. In autoregressive language modeling, many linguistic structures are learned in a nested way. A simple feature is learned first, a composed feature reuses it, and a later feature adds another constraint on top of the previous composition. This structure naturally matches next-token prediction. A model sees a prefix, predicts the next token, then sees a longer prefix, predicts the next token again, and gradually learns a hierarchy of reusable predictive features.

The problem is not compositionality itself. Compositionality is necessary for efficient language learning. The problem is that cross-entropy optimization can attach this compositional reuse to high-gain singular directions. Once a few directions acquire large singular values, later related features are attracted into those directions because a small movement along them produces a large logit or attention-score change, and therefore a larger local loss decrease. The result is a representation space whose nominal dimension may be large, but whose effective dimension is much smaller.

The central conjecture of this agenda is: **\( \text{nested language structure} + \text{next-token prediction} + \text{singular-value amplification} \Rightarrow \text{low-dimensional feature concentration}. \)** Here nested language structure means that complex meanings reuse earlier meanings; next-token prediction means that the training signal repeatedly rewards predictive continuation; singular-value amplification means that directions with large singular values produce larger first-order loss decrease; and low-dimensional feature concentration means that many distinct features live inside a small high-gain subspace.

This agenda asks whether we can preserve the benefit of compositional reuse while preventing singular-value domination. The target is not to destroy linguistic hierarchy. The target is to make the learned representation space use more of its available latent dimension, reduce feature interference, reduce outlier amplification, and improve capacity, especially for long-tail features.

## 1. Starting Point: What The Current Theory Already Gives Us

The existing theory explains why high-gain singular subspaces can attract later compositional features. Let a hidden representation be \(h \in \mathbb{R}^d\), and let an output-facing matrix map it to logits by \(z = Wh\). If the singular value decomposition is \(W = U\Sigma V^\top\), then the hidden-side singular vectors are the columns of \(V\), the output-side singular vectors are the columns of \(U\), and the singular values are the diagonal entries of \(\Sigma\).

If the hidden feature has coefficient \(c_i\) along hidden-side singular vector \(v_i\), then the output effect along the corresponding output direction is amplified by \(\sigma_i\). The basic forward relation is: **\(Wh = \sum_i \sigma_i c_i u_i\).** Here \(h = \sum_i c_i v_i\), \(\sigma_i\) is the gain of singular direction \(i\), and \(u_i\) is the output-side singular vector. This equation is justified directly by the definition of the singular value decomposition.

The loss-descent argument makes the mechanism sharper. For cross-entropy with softmax, the logit gradient is \(\nabla_z L = p - e_y\), where \(p\) is the predicted distribution and \(e_y\) is the one-hot target vector. If we perturb the hidden state by \(\delta h = \epsilon v_i\), then the logit perturbation is \(\delta z = \epsilon \sigma_i u_i\). The first-order loss change is: **\(\delta L \approx \epsilon \sigma_i \langle p-e_y, u_i\rangle\).** Here \(\epsilon\) is the perturbation size, and \(\langle p-e_y, u_i\rangle\) measures whether the output-side singular direction is aligned with the current prediction error. This equation is justified by first-order Taylor expansion of the loss around the current logits.

Therefore, the best local movement along singular direction \(v_i\) gives: **\(\delta L_i^{\star} \approx -\epsilon \sigma_i |\langle p-e_y, u_i\rangle|\).** Here \(\delta L_i^{\star}\) means the most negative first-order loss change achievable by moving only along direction \(v_i\). This equation shows that a large singular value is not automatically useful. It is useful only when the corresponding output-side singular vector has nonzero alignment with the error vector.

The same logic applies to attention. Query and key matrices create attention scores. If a direction in \(W_Q\) or \(W_K\) creates a useful score gap, softmax converts that score gap into a routing preference. The previous article explains this effect locally: high-gain Q/K directions are not merely amplified after being chosen; they are chosen because movements along them can produce larger score changes and therefore larger first-order loss decreases when the score-gradient aligns with the useful routing error.

So the existing mechanism is not simply a geometric observation. It is an optimization argument. High-singular-value directions are reused because they are efficient descent directions.

## 2. The New Problem: The Model Has Large Dimension But Small Effective Space

A transformer may have a large hidden dimension \(d\), but the learned features may occupy only a much smaller effective subspace. This motivates the effective-dimension diagnosis: **\(d_{\mathrm{eff}} \ll d\).** Here \(d\) is the nominal architectural dimension, while \(d_{\mathrm{eff}}\) is the effective dimension used by learned features or gradients. This statement is a measurable conjecture, not a definition. It should be tested by spectral effective rank, feature covariance rank, gradient covariance rank, and causal intervention.

One possible measurement is the covariance effective dimension: **\(d_{\mathrm{eff}}(\Sigma_h) = \frac{(\operatorname{tr}\Sigma_h)^2}{\operatorname{tr}(\Sigma_h^2)}\).** Here \(\Sigma_h\) is the covariance matrix of hidden representations, \(\operatorname{tr}\Sigma_h\) is the total variance, and \(\operatorname{tr}(\Sigma_h^2)\) measures how concentrated that variance is across principal directions. This formula is justified by the participation ratio: it equals the number of equally sized dimensions that would produce the same concentration level.

A second measurement is singular effective rank for a matrix \(W\): **\(r_{\mathrm{eff}}(W) = \exp\left(-\sum_i \rho_i \log \rho_i\right), \quad \rho_i = \frac{\sigma_i}{\sum_j \sigma_j}\).** Here \(\sigma_i\) is the \(i\)-th singular value, \(\rho_i\) is the normalized spectral mass, and \(r_{\mathrm{eff}}\) is the entropy effective rank. This formula is justified by the entropy of the singular-value distribution: it is large when spectral mass is spread and small when spectral mass is concentrated.

A third measurement is feature top-subspace mass: **\(M_k(u) = \sum_{i=1}^k \langle u, v_i\rangle^2\).** Here \(u\) is a learned feature direction, and \(v_i\) is the \(i\)-th hidden-side singular vector of a model matrix. This quantity measures how much of the feature lies in the top \(k\) singular directions. It is justified by orthogonal projection: the squared projection mass onto a subspace is exactly the sum of squared inner products with an orthonormal basis of that subspace.

The empirical hypothesis is that ordinary cross-entropy training creates a gap between architectural dimension and effective dimension. The model has the capacity to use many dimensions, but the optimization dynamics concentrate predictive learning into a small number of high-gain directions.

## 3. Why Nested Language Makes This Problem Natural

Language is not a flat set of independent labels. It is nested, compositional, and recursively reusable. A phrase such as “city” can become “beautiful city,” then “NYC is a beautiful city,” then “NYC is a beautiful city in winter.” Each longer phrase preserves some earlier semantic content and adds a new constraint.

The same structure appears in synthetic form as \(A\), \(AB\), \(ABC\), and \(ABCD\). The model first learns that \(A\) predicts \(B\). Then \(AB\) predicts \(C\). Then \(ABC\) predicts \(D\). This is not a defect of the dataset. It isolates a genuine feature of language: later predictions reuse earlier predictive structure.

The central mechanism can be stated as: **\(u_{ABC} \approx u_{AB} + \Delta_{AB\to ABC}\).** Here \(u_{ABC}\) is the representation direction for the longer prefix, \(u_{AB}\) is the inherited representation direction, and \(\Delta_{AB\to ABC}\) is the newly added feature. This equation is not assumed to be exact. It is a modeling decomposition used to define inherited and extra components for measurement.

The problem appears when many different \(\Delta\) components are distinct in feature space but still live inside the same high-gain singular subspace. In that case, the model has learned multiple features, but the features are coupled through the same amplified directions.

This gives the main causal chain: **\(\text{frequent inherited structure} \to \text{large singular values} \to \text{later-feature attraction} \to \text{shared subspace} \to \text{interference}.\)** Here “frequent inherited structure” means that common compositional patterns receive many updates; “large singular values” means those updates create high-gain directions; “later-feature attraction” means later features receive larger descent benefit from those directions; “shared subspace” means many features occupy the same top singular directions; and “interference” means that updating one feature changes other features that share those directions.

## 4. The Core Research Question

The next research question should be stated precisely:

Can we train language models so that compositional features still reuse useful structure, but the representation space does not collapse into a small number of singular-value-dominated directions?

This question has two parts. The first part is diagnostic: does ordinary training actually under-utilize the representation space in a way that can be measured and causally linked to interference? The second part is constructive: can we change the training geometry, representation geometry, or routing architecture so that the model uses more of the latent space without sacrificing prediction quality?

The desired representation geometry is: **\(d_{\mathrm{eff}} \approx d\), while \(L_{\mathrm{CE}}\) remains low.** Here \(d_{\mathrm{eff}} \approx d\) means the learned features use most of the available dimension, and low \(L_{\mathrm{CE}}\) means the model still solves next-token prediction. This statement is justified as an objective criterion: the method should not merely make representations isotropic by destroying useful prediction.

The undesired geometry is: **\(d_{\mathrm{eff}} \ll d\), even when prediction accuracy is high.** Here the model may appear successful by next-token loss, but its internal feature space is fragile because many features are packed into a small high-gain subspace. This statement is falsifiable by measuring whether effective dimension remains small after successful training and whether interventions on top singular directions damage many unrelated features.

## 5. Conjecture 1: Singular-Value Domination Reduces Usable Capacity

The first conjecture is: **\(\Delta L_i \propto -\sigma_i^2 a_i^2\).** Here \(\Delta L_i\) is the first-order loss decrease from a gradient step restricted to singular direction \(i\), \(\sigma_i\) is the singular value, and \(a_i = \langle u_i, p-e_y\rangle\) is the alignment between the output-side singular vector and the prediction error. This conjecture follows from the local loss-descent derivation in the current theory.

The implication is that optimization does not treat all semantically useful directions equally. If two directions have similar error alignment but one has larger \(\sigma_i\), the high-gain direction gives a larger local descent benefit. Over many training steps, this creates a self-reinforcing process: a direction becomes useful, its singular value grows, and its larger singular value makes it even more attractive for future updates.

The profiling experiment is straightforward. During training, periodically compute singular spectra of \(W_{\mathrm{out}}\), \(W_Q\), \(W_K\), \(W_V\), and MLP matrices. At the same checkpoints, compute feature directions for controlled patterns and measure their top-subspace mass \(M_k(u)\). The prediction is that feature top-subspace mass will increase together with spectral concentration in output-facing and routing-facing matrices.

The falsification criterion is also clear. If singular values become large but feature directions do not concentrate into the corresponding singular subspaces, then singular-value domination is not the main cause of feature concentration. If feature concentration appears before singular concentration, then the causal order must be revised. If causal intervention on top singular directions does not damage many learned features, then the alleged shared-subspace coupling is weak.

The updated conjecture would then need to distinguish spectral concentration from functional concentration. It may be that some singular concentration is harmless if the large directions are not used by many features, or if downstream normalization removes their functional effect.

## 6. Conjecture 2: A Good Representation Space Should Be Distributed But Not Structureless

The second conjecture is: **\(\text{good representation} \neq \text{maximally isotropic noise}.\)** A good representation space should use many dimensions, but it should not erase semantic hierarchy. This distinction matters because language is genuinely compositional. If we force all features to be orthogonal or uniformly distributed without regard to meaning, we may destroy useful reuse.

A better target is: **\(\text{distributed capacity} + \text{controlled reuse}.\)** Here distributed capacity means that features occupy many dimensions, while controlled reuse means semantically related features may still share subspaces when sharing helps prediction. This statement is a design principle rather than a theorem, and it must be justified experimentally by comparing loss, generalization, and interference.

The measurable version is: **\(M_k(u_{\mathrm{related}}) > M_k(u_{\mathrm{unrelated}})\), but no small \(k\) explains most features.** Here related features are features that should share structure, unrelated features are features that should not interfere, and \(M_k\) measures top-subspace mass. This criterion says that semantic sharing should exist, but it should not become global collapse.

The profiling experiment should therefore separate inherited structure, extra structure, and unrelated structure. In the nested-prefix case, inherited structure is measured by the projection of \(u_Y\) onto \(u_X\), extra structure is measured by \(u_Y - (u_Y^\top u_X)u_X\), and unrelated structure is measured using random or semantically unrelated prefix families. If a method works, inherited components may share subspace, but extra and unrelated components should have more distributed support.

The falsification criterion is that a proposed method increases effective dimension only by spreading meaningless noise. If effective dimension increases but causal feature probes become worse, the representation is not better. It is merely less concentrated.

## 7. Solution Family A: Spectral Control

The most direct intervention is to control singular values. The idea is to prevent any small group of directions from becoming overwhelmingly high-gain. This does not remove compositional learning. It reduces the optimization advantage of already-amplified directions.

A simple objective is: **\(\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \lambda \sum_i (\sigma_i - \bar{\sigma})^2\).** Here \(\mathcal{L}_{\mathrm{CE}}\) is the ordinary cross-entropy loss, \(\lambda\) is the regularization strength, \(\sigma_i\) is a singular value of a selected matrix, and \(\bar{\sigma}\) is a target average singular value. This objective is justified by the desire to reduce spectral inequality directly.

A weaker version is spectral clipping: **\(\sigma_i \leftarrow \min(\sigma_i, \tau)\).** Here \(\tau\) is a maximum allowed singular value. This operation prevents extreme outliers but does not force all singular values to be equal. It is easier to implement approximately through spectral normalization or power-iteration constraints.

A more adaptive version is gain equalization: **\(\widehat{W} = W( W^\top W + \epsilon I)^{-1/2}\).** Here \(W\) is the original matrix, \(\widehat{W}\) is a whitened or approximately orthogonalized version, \(\epsilon I\) stabilizes the inverse square root, and the operation reduces anisotropic gain on the hidden side. This equation is justified by the fact that \(\widehat{W}^\top\widehat{W}\) becomes close to identity when \(\epsilon\) is small and \(W^\top W\) is well conditioned.

The testable implication is that spectral control should reduce \(M_k(u)\) for extra features, increase \(d_{\mathrm{eff}}\), reduce spectral outliers, and preserve structured prediction accuracy. The risk is underfitting. Some high-gain directions may be genuinely useful for common structure. If spectral control is too strong, the model may lose efficient compositional reuse.

The falsification criterion is that spectral control reduces singular concentration but worsens next-token loss or destroys compositional continuation accuracy. In that case, singular concentration is not merely pathological; it is also carrying useful computation that must be replaced rather than suppressed.

## 8. Solution Family B: Scale-Free or Directional Representation Learning

The second intervention is to reduce the role of magnitude in representation learning. The key idea is that features should be compared by direction or relational structure rather than by amplified norm. This is often described loosely as scale-free learning, but the precise design matters.

A normalized directional representation is: **\(\tilde{h} = \frac{h}{\|h\| + \epsilon}\).** Here \(h\) is the hidden representation, \(\tilde{h}\) is the normalized representation, and \(\epsilon\) avoids division by zero. This removes the ability of hidden norm alone to dominate logits or attention scores.

A normalized output classifier is: **\(z_y = \tau \left\langle \frac{w_y}{\|w_y\|+\epsilon}, \frac{h}{\|h\|+\epsilon} \right\rangle\).** Here \(z_y\) is the logit for token \(y\), \(w_y\) is the output vector for token \(y\), \(h\) is the hidden state, and \(\tau\) is a temperature or learned scale. This equation is justified by cosine classification: prediction depends on angular alignment rather than unbounded vector norms.

For attention, the corresponding normalized score is: **\(s_{ti} = \tau_q \left\langle \frac{q_t}{\|q_t\|+\epsilon}, \frac{k_i}{\|k_i\|+\epsilon} \right\rangle\).** Here \(q_t\) is the query, \(k_i\) is the key, and \(\tau_q\) controls attention sharpness. This reduces the ability of large Q/K norms or singular gains to dominate routing.

Hyperbolic representation learning is related but should be treated carefully. Hyperbolic space is useful for hierarchical data because distance can grow exponentially with radius, but it is not automatically scale-free. In hyperbolic geometry, radial position is meaningful. Therefore, the relevant conjecture is not simply “use hyperbolic space.” The relevant conjecture is that a geometry with explicit hierarchy and controlled scale may represent nested language structure without requiring uncontrolled Euclidean singular-value amplification.

The testable implication is that directional or controlled-geometry training should reduce outlier singular values, reduce logit outliers, increase representation effective dimension, and preserve compositional prediction. The risk is that removing magnitude may reduce the model’s ability to express confidence. This risk can be managed by separating semantic direction from confidence scale, using a learned global or local temperature.

The falsification criterion is that normalized training reduces outliers but increases loss significantly or weakens rare-feature learning. That outcome would mean that magnitude is not merely a nuisance; it is part of the model’s useful confidence mechanism. The theory would then need to distinguish harmful singular amplification from necessary confidence scaling.

## 9. Solution Family C: Expert-Partitioned Representation Spaces

The third intervention is to stop forcing all features through the same shared high-gain matrices. Instead, the model can partition the representation space into many expert subspaces. This changes the problem from one global subspace absorbing all structure to many local subspaces specializing in different regions of semantic space.

The basic expert routing function is: **\(g(h): \mathbb{R}^d \to \{1,\ldots,E\}\).** Here \(g(h)\) maps a token representation \(h\) to one of \(E\) experts. This equation defines the routing problem. It does not yet solve it.

If \(E = 10^6\), then the number of binary decisions needed to index an expert is approximately: **\(\log_2 E \approx 20\).** Here \(E\) is the number of experts, and \(\log_2 E\) is the number of bits needed to uniquely address one expert. This justifies the intuition that one million experts do not require a million-dimensional gate. In principle, a 20-bit code is enough to address them.

The real problem is semantic stability. A compact code can address many experts, but arbitrary codes are useless. The gate must map related features to related experts and conflicting features to separated experts. A better formulation is: **\(b(h) \in \{0,1\}^m, \quad E = 2^m\).** Here \(b(h)\) is an \(m\)-bit semantic routing code, and \(2^m\) is the number of addressable experts. This equation is justified by binary coding, but the learning problem is to make the code semantically meaningful.

The desired routing property is: **\(\operatorname{dist}(b(h_a), b(h_b)) \approx \operatorname{dist}_{\mathrm{sem}}(a,b)\).** Here \(\operatorname{dist}(b(h_a), b(h_b))\) is the Hamming or tree distance between routing codes, and \(\operatorname{dist}_{\mathrm{sem}}(a,b)\) is a task-defined semantic distance between features. This is a conjectural objective, not a given property. It must be learned and tested.

The expert-partitioned view also suggests a different role for singular vectors. Instead of allowing a few global singular directions to dominate all features, each expert can own a local singular subspace. The global model becomes a collection of locally structured spaces rather than one globally collapsed space.

The testable implication is that expert partitioning should reduce cross-feature interference, especially between common and rare features. Rare features should not have to fight inside the same high-gain subspace dominated by common features. The risk is load imbalance and routing collapse. If the gate sends too many tokens to a few experts, the same collapse reappears at the expert level.

The falsification criterion is that the MoE model increases parameter count but does not increase effective feature diversity, or that routing entropy looks high while causal feature specialization remains weak. In that case, the model is distributing computation superficially but not solving representational collapse.

## 10. A Concrete Gating Research Program

The gating problem should not be treated as a minor engineering detail. It is the core mathematical problem of the expert-partitioned solution.

A useful gate should satisfy three constraints. First, it should be balanced, so experts receive enough data. Second, it should be semantic, so related features route to related experts. Third, it should be stable, so small irrelevant perturbations do not cause arbitrary expert changes.

The balance objective can be written as: **\(\mathcal{L}_{\mathrm{bal}} = \sum_{e=1}^E \left(\hat{p}_e - \frac{1}{E}\right)^2\).** Here \(\hat{p}_e\) is the empirical fraction of tokens routed to expert \(e\), and \(1/E\) is the ideal uniform fraction. This objective is justified by load balancing: it penalizes routing collapse into a small number of experts.

The semantic consistency objective can be written as: **\(\mathcal{L}_{\mathrm{sem}} = \mathbb{E}_{(a,b)} \left[\left(\operatorname{dist}(b(h_a), b(h_b)) - \alpha\operatorname{dist}_{\mathrm{sem}}(a,b)\right)^2\right]\).** Here \((a,b)\) is a pair of features or examples, \(b(h_a)\) and \(b(h_b)\) are their routing codes, \(\operatorname{dist}_{\mathrm{sem}}\) is a semantic distance obtained from controlled data or probes, and \(\alpha\) rescales semantic distance to code distance. This objective is justified by the requirement that routing geometry should reflect semantic geometry.

The stability objective can be written as: **\(\mathcal{L}_{\mathrm{stab}} = \mathbb{E}_{h,\xi}\left[\operatorname{dist}(b(h), b(h+\xi))\right]\).** Here \(\xi\) is a small perturbation that should not change the semantic identity of the token. This objective is justified by robustness: irrelevant noise should not produce unstable routing.

A practical training objective is: **\(\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{bal}}\mathcal{L}_{\mathrm{bal}} + \lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}} + \lambda_{\mathrm{stab}}\mathcal{L}_{\mathrm{stab}}\).** Here the \(\lambda\) coefficients control how strongly the model enforces balance, semantic consistency, and stability. This objective is justified as a multi-constraint implementation of the gating desiderata.

The open question is whether semantic routing can be learned from next-token supervision alone, or whether it requires auxiliary structure. This should be tested rather than assumed. Controlled synthetic data can provide ground-truth semantic distances. Natural language can then use approximations from phrase templates, topic labels, retrieval clusters, or causal probes.

## 11. Main Experimental Plan

The experimental program should proceed in stages. The first stage should stay close to the current nested-prefix setup. The reason is methodological: before testing natural language, we need a setting where the inherited feature, extra feature, frequency, and conflict structure are all known.

The baseline dataset contains nested sequences such as \(A\), \(AB\), \(ABC\), \(ABCD\), and longer continuations. A conflict-augmented dataset should add cases where common and rare features share inherited structure but require different later predictions. This isolates whether high-gain reuse helps common patterns while hurting rare patterns.

The model set should include a standard transformer, a spectral-control transformer, a directional-normalized transformer, and an expert-partitioned transformer. The comparison should not only report final validation loss. It should report spectral geometry, representation geometry, routing geometry, and causal interference.

The primary metrics are: cross-entropy loss, structured continuation accuracy, rare-feature accuracy, singular effective rank, representation effective dimension, feature top-subspace mass, gradient covariance effective dimension, routing entropy, expert load balance, and causal damage under subspace ablation.

The most important causal test is top-subspace removal. Let a feature representation be decomposed as: **\(h_f = h_{f,\mathrm{top}} + h_{f,\mathrm{tail}}\).** Here \(h_{f,\mathrm{top}}\) is the projection onto the top singular subspace, and \(h_{f,\mathrm{tail}}\) is the remaining component. This decomposition is justified by orthogonal projection onto the singular-vector basis.

The intervention is: **\(h_f' = h_f - \lambda h_{f,\mathrm{top}}\).** Here \(\lambda\) controls how much of the top-subspace component is removed. If many features fail together when \(h_{f,\mathrm{top}}\) is removed, then they are functionally coupled through the same high-gain subspace.

The complementary intervention is: **\(h_f' = h_f - \lambda h_{f,\mathrm{tail}}\).** Here the tail component is removed instead. If rare features depend more on the tail component than common features, then the tail subspace may carry specificity while the top subspace carries shared continuation structure.

The success criterion is not just better loss. The success criterion is: **\(L_{\mathrm{CE}} \downarrow, \quad d_{\mathrm{eff}} \uparrow, \quad M_k(u_{\mathrm{extra}}) \downarrow, \quad \text{interference} \downarrow.\)** Here lower cross-entropy means the model still predicts well, higher effective dimension means the representation uses more space, lower extra-feature top-subspace mass means new features are less collapsed, and lower interference means training or ablating one feature damages fewer unrelated features.

## 12. Natural-Language Scaling Experiments

After the controlled setting, the same agenda should move to semi-natural phrases. The goal is not to prove all language semantics at once. The goal is to test whether the same mechanism appears when symbolic prefixes are replaced by phrase-level composition.

A semi-natural dataset can use templates such as “city,” “beautiful city,” “NYC is a beautiful city,” and “NYC is a beautiful city in winter.” Another family can use factual compositions such as “capital of France,” “capital of France is Paris,” and “the old capital of France was ...” The templates should explicitly separate inherited components from newly added constraints.

The representation decomposition remains the same: **\(u_{\mathrm{extra}}(X\to Y) = \operatorname{normalize}(u_Y - (u_Y^\top u_X)u_X)\).** Here \(u_X\) is the feature direction for the shorter phrase, \(u_Y\) is the feature direction for the longer phrase, and the subtraction removes the inherited component. This formula is justified by projecting \(u_Y\) away from \(u_X\) to isolate the newly added information.

The natural-language prediction is that ordinary cross-entropy training will again concentrate many extra features into top singular subspaces of output-facing and routing-facing matrices. Spectral control, directional normalization, or expert partitioning should reduce this concentration while preserving phrase-continuation accuracy.

The falsification criterion is that natural phrases do not show the same singular concentration pattern, even when controlled synthetic data does. That would limit the theory. It would mean the current mechanism is real but not necessarily dominant in natural-language composition.

## 13. Why This Could Improve Parallel Training and Inference

If representation space is more distributed and expert routing is more precise, the architecture becomes naturally parallel. Different semantic regions can be processed by different experts, and the gate becomes an address function rather than a small top-\(k\) heuristic.

The parallelism conjecture is: **\(\text{semantic partitioning} \Rightarrow \text{lower interference} + \text{higher parallel efficiency}.\)** Here semantic partitioning means that tokens are routed into specialized subspaces, lower interference means fewer unrelated features share the same high-gain directions, and higher parallel efficiency means computation can be distributed across experts with less redundant activation.

This is not automatic. If the gate is unstable or imbalanced, parallelism becomes inefficient. If the experts are too small, they may lose capacity. If experts are too numerous, routing overhead may dominate. Therefore, the systems question must be integrated with the representation question.

The relevant systems metric is not just FLOPs. It is useful activated capacity per token. A dense model activates the same global parameter space for every token. A well-routed expert model activates a smaller but more relevant subspace. The hypothesis is that better representation partitioning can increase effective capacity without proportional inference cost.

## 14. Expected Outcomes If The Agenda Works

If the agenda works, several outcomes should appear together. Singular spectra should become less extreme. Hidden representation covariance should have higher effective dimension. Extra compositional features should occupy broader or more semantically partitioned subspaces. Rare features should become less fragile. Causal removal of top singular directions should damage fewer unrelated features. Expert routing should become more stable and semantically meaningful. Training and inference should become more parallelizable because computation is distributed across specialized subspaces.

The strongest version of the claim is: **\(\text{scale-controlled distributed representation} \Rightarrow \text{higher semantic capacity at fixed model size}.\)** Here scale-controlled means singular-value domination is reduced, distributed representation means features use more of the latent space, and higher semantic capacity means the model can store and retrieve more distinct features with less interference. This claim is ambitious and must be tested by controlled scaling laws.

The weaker but still valuable claim is: spectral and routing interventions reduce pathological outliers and improve long-tail feature robustness without harming common-feature prediction. This would already be significant, because it would connect representation geometry, singular-value dynamics, and long-tail reliability.

## 15. Main Risks and Alternative Explanations

The first risk is that singular concentration may be an efficient compression mechanism rather than a pathology. If so, reducing it may hurt loss. The response is not to reject the theory, but to refine it: some concentration is useful, but excessive global concentration causes interference.

The second risk is that apparent low-rank structure may be a measurement artifact. Hidden states may look low-dimensional under one metric but still encode information nonlinearly. The response is to use causal interventions, not just spectral measurements.

The third risk is that expert partitioning may move the problem rather than solve it. A model with many experts can still collapse if the gate routes most tokens to a few experts or if each expert internally develops its own high-gain collapse. The response is to measure both global and per-expert effective dimension.

The fourth risk is that scale-free or normalized training may reduce confidence calibration. The response is to separate semantic direction from confidence scale, rather than removing scale entirely.

## 16. Final Research Thesis

The proposed thesis is:

**\(\text{Large language models may be architecturally high-dimensional but dynamically low-dimensional.}\)** Here architecturally high-dimensional means the model has a large hidden space and many parameters, while dynamically low-dimensional means the training process concentrates many useful features into a small number of high-gain directions.

The cause is not merely bad optimization. The cause may be the interaction between nested human language, next-token prediction, softmax cross-entropy, and singular-value amplification. This interaction makes feature reuse efficient, but it can also create feature crowding.

The constructive goal is:

**\(\text{preserve compositional reuse while preventing singular-value domination.}\)** Here preserving compositional reuse means the model should still exploit the nested structure of language, while preventing singular-value domination means no small set of amplified directions should monopolize the learning dynamics.

The research program is therefore to design, train, and test models with more scale-controlled, distributed, and semantically partitioned representation spaces. If successful, such models should have fewer spectral outliers, higher effective dimension, lower feature interference, better long-tail robustness, and more natural parallelism for training and inference.

