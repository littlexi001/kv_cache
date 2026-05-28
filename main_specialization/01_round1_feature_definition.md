# Round1 Feature Definition: Naive Synthetic Data

第一轮 synthetic 数据采用最干净的层次化构造：我们预先定义一个 token pool，再由若干 token 构成 local slot，由若干 local slot 构成 higher-level slot。不同 local slot 之间不共享 token，不同 higher-level slot 之间也不共享 local slot。

在这个设定下，每个 token 本身可以被视为一个独立 feature；同一 local slot 内的 token 共享一个 local-level feature；同一 higher-level slot 内的 local slots 共享一个更高层次的 compositional feature。因此，feature 的 ground truth 完全由数据生成规则给出。

对应地，specialization 可以被形式化为：对于属于同一个 ground-truth feature group 的 token / position，MoE gate 应当把它们分发到相同或高度重叠的 expert bucket；对于属于不同 feature group 的 token / position，MoE gate 应当产生可区分的 expert assignment。

这一设定的优点是 ground truth 清晰，可以直接计算 local slot / higher-level slot 与 expert assignment 的一致性，例如 feature-to-expert purity、same-feature same-expert rate、MI / NMI 等指标。

它的缺点也很明确：它过于干净，弱化了真实语言中一词多义、同义改写和上下文依赖带来的歧义。
