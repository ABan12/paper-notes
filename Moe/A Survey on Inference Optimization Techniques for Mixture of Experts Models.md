**《A Survey on Inference Optimization Techniques for Mixture of Experts Models》**
---

# 1. 论文内容

这篇论文是一篇 **MoE 模型推理优化综述**。

1. **模型层优化**：改 MoE 架构、剪枝、量化、蒸馏、路由优化等。
2. **系统层优化**：多 GPU/多机并行、专家并行、专家卸载、缓存、预取等。
3. **硬件层优化**：FPGA、近数据计算、PIM、MoE 专用加速器等。

它的核心观点是：
**MoE 模型虽然只激活一小部分专家，看起来计算量低，但推理部署并不简单，因为专家选择是动态的，会带来负载不均衡、通信开销、显存压力、专家加载延迟等问题。**

论文地址：[https://arxiv.org/abs/2412.14219]

---

# 2. 为什么 MoE 推理优化很重要？

传统 dense LLM 是所有参数都参与计算。
MoE 的想法是：模型里有很多个专家，但每个 token 只激活其中几个专家。

例如一个 MoE FFN 层可以有 8 个专家，但每个 token 只选 top-2 个专家。这样模型总参数很多，但实际计算参数较少。

论文中提到，MoE 的优势主要有三个：

第一，**条件计算**。
不是所有参数都算，而是按输入选择专家，因此可以用较少计算量获得较大模型容量。

第二，**专家 specialization**。
不同专家可以学习不同类型的知识，比如代码、数学、某些语义模式、某些语言现象。

第三，**动态路由**。
每个 token 经过 router，选择最合适的专家。复杂 token 可以调用更多或更合适的专家，简单 token 可以少算。

但是问题也随之出现：

MoE 的瓶颈往往不只是 FLOPs，而是：

* 专家参数太多，显存放不下；
* 不同 token 选择不同专家，导致调度复杂；
* 某些专家特别热门，某些专家没人用，导致负载不均衡；
* 多 GPU 时需要 all-to-all 通信；
* 边缘设备上需要专家卸载，专家加载成为瓶颈；
* 当前硬件主要为 dense computation 优化，不适合动态稀疏 MoE。

---

# 3. MoE 的基本推理流程

论文先介绍 MoE 的基本计算方式。

一个 MoE 层一般包含：

* 一个 router / gate；
* 多个 expert；
* 一个 top-k 选择机制；
* 一个加权聚合机制。

流程大概是：

## 第一步：router 给每个专家打分

输入 token 表示为 (x)。
router 计算每个专家的概率：

[
\theta = Softmax(R(x))
]

这里 (\theta) 是每个专家的选择概率。

## 第二步：选择 top-k 专家

[
E_{selected} = TopK(\theta, K)
]

比如总共有 8 个专家，只选 top-2。

## 第三步：被选中的专家并行计算

[
y_i = E_i(x)
]

只有被选中的专家参与计算。

## 第四步：加权合并专家输出

[
y = \sum_{i \in E_{selected}} \frac{\theta_i}{\sum_{j \in E_{selected}}\theta_j} y_i
]

也就是说，专家输出不是简单平均，而是按照 router 给出的概率加权。

你可以把 MoE 想成一个“动态分流系统”：

> router 像调度员，token 像请求，expert 像专业服务窗口。每个 token 来了以后，router 决定它去哪些窗口处理，最后把几个窗口的结果合并。

---

# 4. 论文的核心分类框架

这篇论文最重要的贡献是把 MoE 推理优化整理成一个 taxonomy。

它分成三层：

| 层级  | 解决什么问题                  | 代表技术                          |
| --- | ----------------------- | ----------------------------- |
| 模型层 | 从模型结构本身减少计算和参数          | 剪枝、量化、蒸馏、动态 gating、专家合并       |
| 系统层 | 让 MoE 在 GPU/CPU/集群上跑得更快 | 专家并行、all-to-all 优化、专家卸载、缓存、预取 |
| 硬件层 | 设计更适合 MoE 的硬件           | FPGA、PIM、近数据计算、专用加速器          |

对于你的课题“**MoE 模型剪枝**”，最相关的是 **Section 3.2 Model Compression Techniques**，尤其是 **3.2.1 Expert Pruning**。

---

# 5. 模型层优化：这部分和你最相关

模型层优化分成三类：

1. 架构设计；
2. 模型压缩；
3. 算法改进。

---

## 5.1 架构设计：让 MoE 模型本身更高效

论文把 MoE 架构设计分成两类：

### A. MoE-based Attention

传统 MoE 多数是替换 Transformer 里的 FFN，也就是把 FFN 变成多个专家。
但一些工作开始把 attention 也做成 MoE。

例如：

* MoA：Mixture of Attention Heads；
* SwitchHead；
* MoH；
* JetMoE；
* ModuleFormer。

这些方法的核心思想是：

> 不只是 FFN 可以稀疏化，attention head 也可以按 token 动态选择。

这类工作对推理优化的意义是：
如果 attention 也能稀疏激活，那么不仅 FFN 计算可以减少，attention 计算也可以减少。

不过对你的“剪枝”课题而言，这部分不是最核心，可以先了解即可。

---

### B. MoE-based FFN

大多数 LLM MoE 的专家其实就是 FFN expert。
所以优化 FFN MoE 是最主流方向。

论文提到的代表工作包括：

* MoE++：引入 zero-computation experts，减少计算；
* Pre-gated MoE：提前预测下一层需要哪些专家，方便预取；
* SCoMoE：优化通信；
* COMET：用树结构优化专家选择；
* MoELoRA：把 LoRA 也设计成 MoE 风格。

这部分说明一个趋势：

> MoE 优化越来越不是单纯改模型，而是模型设计和系统执行一起考虑。

---

## 5.2 模型压缩：剪枝、量化、蒸馏、分解

这是论文和你课题最相关的一章。

论文指出，MoE 模型中大部分参数都在专家里。
例如 Mixtral-8x7B 中，专家参数占比非常高。因此压缩 MoE 时，通常优先压缩专家。

模型压缩主要分四类：

1. Expert Pruning：专家剪枝；
2. Expert Quantization：专家量化；
3. Expert Distillation：专家蒸馏；
4. Expert Decomposition：专家低秩分解。

---

# 6. Expert Pruning：你的重点方向

论文把专家剪枝分成两大类：

## 6.1 Structured Expert Pruning：结构化专家剪枝

这类方法直接减少专家数量。

比如原来每层有 8 个专家，剪掉一部分后只保留 4 个专家。

优点：

* 显存直接下降；
* 推理时可选专家变少；
* 系统部署更简单；
* 对硬件友好。

缺点：

* 容易损失模型能力；
* 某些专家可能只在特定任务上重要，盲目剪掉会伤害泛化；
* 每层剪多少、剪哪些专家并不容易。

论文中提到的代表方法包括：

### TSEP

Task-Specific Expert Pruning。
它针对下游任务剪掉“不专业”的专家，只保留对目标任务有用的专家。

适合场景：

> 你只关心某个特定任务，比如数学、代码、翻译、分类，那么可以剪掉对这个任务不重要的专家。

它的核心是 task-specific。

---

### NAEE

Not All Experts Are Equal。
它认为不同专家重要性不同，用小校准集评估专家组合，然后删除不重要专家。

这类方法的核心是：

> 用 calibration dataset 判断专家重要性，而不是只看参数大小。

---

### UNCURL

它根据 MoE router logits 来减少专家。
也就是说，不只是看专家本身，而是看 router 对专家的选择情况。

这对剪枝非常关键，因为在 MoE 中，一个专家“重要不重要”往往不是由参数范数决定，而是由它在真实数据中是否经常被 router 选中、选中时贡献多大决定。

---

### SEER-MoE

它使用 heavy-hitters counting 方法统计常用专家，然后结合正则化微调进一步剪枝。

可以理解为：

> 经常被选中的专家更可能重要，很少被选中的专家更可能冗余。

但这也有风险：
低频专家可能负责长尾能力，剪掉后可能损害某些罕见任务。

---

### MoE-Pruner

这是比较重要的一个方向。
它做的是 unstructured pruning，也就是剪 expert 内部权重，而不是直接删除整个专家。

它用：

* 权重大小；
* 输入激活；
* router 权重；

共同判断参数重要性。

这比普通 magnitude pruning 更适合 MoE，因为 MoE 参数是否重要不仅取决于权重本身，还取决于该 expert 是否被激活，以及激活时 router 给多大权重。

---

### STUN

Structured-Then-Unstructured Pruning。
先做结构化剪枝，再做非结构化剪枝。

这个思路很合理：

第一步先减少专家数量；
第二步再压缩保留下来的专家内部参数。

这样可以兼顾：

* 粗粒度压缩；
* 细粒度压缩；
* 推理速度；
* 模型精度。

---

### MoE-Compression

论文说它提出一个统一框架，同时使用 structured pruning 和 unstructured pruning，在较小精度损失下获得显著推理加速。

这类工作适合你作为系统性剪枝框架的参考。

---

## 6.2 Expert Merging：专家合并

除了直接删除专家，论文还总结了另一类方法：**合并专家**。

比如一层有 8 个专家，发现其中一些专家行为相似，就把它们合并成一个专家。

优点：

* 比直接删除更温和；
* 可以保留多个专家的知识；
* 对精度更友好。

缺点：

* 合并后的专家可能不再专精；
* 合并策略复杂；
* 可能需要微调恢复性能。

代表方法包括：

### MC-SMoE

根据 routing policy 把专家分组，然后每组专家合并成一个专家。

它的逻辑是：

> 如果一些专家经常服务相似的 token 或者路由模式相近，那么它们可能可以合并。

---

### HC-SMoE

使用层次聚类做专家合并，而且不需要重新训练。

这是一个很值得关注的方向，因为 retraining-free 对真实部署很有价值。

---

### DEK

先在特征空间里找相似专家，再在权重空间里合并专家。

它比单纯看权重距离更合理，因为两个专家参数不一定接近，但功能可能类似；反过来，参数接近也不一定代表行为类似。

---

### EEP

同时做 prune 和 merge，既减少总专家数量，也减少 active experts 数量。

这对推理优化更直接，因为 MoE 推理成本不仅来自总参数，也来自每个 token 激活多少专家。

---

### LiteMoE

面向移动设备部署。
它保留最关键专家，合并次要专家，并且不需要重新训练。

这个方向和 edge inference 很相关。

---

# 7. 专家剪枝方法之间怎么比较？

论文中的表 2 比较了多个剪枝方法，比较维度包括：

* 最大剪枝率；
* 是 task-specific 还是 task-agnostic；
* 是否需要微调；
* 是结构化剪枝还是非结构化剪枝；
* 是删除专家还是合并专家。

你可以这样理解：

| 类型                        | 核心问题           | 适合场景      |
| ------------------------- | -------------- | --------- |
| Task-specific pruning     | 针对某个任务剪专家      | 下游任务固定    |
| Task-agnostic pruning     | 不针对具体任务，保留通用能力 | 通用 LLM 部署 |
| Delete experts            | 直接删专家          | 追求简单、高效   |
| Merge experts             | 合并专家           | 希望减少精度损失  |
| Unstructured pruning      | 剪专家内部权重        | 需要更细粒度压缩  |
| Structured + unstructured | 先删专家再剪权重       | 高压缩率场景    |

对你的研究而言，一个很自然的问题是：

> 如何判断一个专家“真的不重要”？

可以从以下几个角度设计指标：

1. **使用频率**：该专家被 router 选中的次数；
2. **router 权重**：被选中时 gate score 是否高；
3. **输出贡献**：去掉它后输出变化多大；
4. **任务敏感性**：去掉它后下游任务损失多少；
5. **专家相似度**：是否和其他专家功能重复；
6. **层间差异**：浅层专家和深层专家的重要性可能不同；
7. **token 类型差异**：专家可能只对某类 token 重要。

---

# 8. 量化：另一种专家压缩方法

论文还总结了 expert quantization。

量化不是删除专家，而是降低专家权重精度。
比如 FP16 → INT8 → INT4 → INT2 → 甚至 1 bit。

代表方法包括：

* MC-MoE；
* MoQE；
* QMoE；
* CMoE；
* HOBBIT；
* EdgeMoE；
* QMoE-Benchmark。

论文指出，量化通常可以显著减少显存，比如 4x 甚至更高，但速度提升不一定稳定。

原因是：

> 低比特权重不等于自动加速，还需要对应 kernel 和硬件支持。

例如 QMoE 虽然压缩很激进，但如果没有高效 1-bit CUDA kernel，可能反而没有速度优势。

对你的剪枝课题来说，量化可以作为组合方向：

> 重要专家保留高精度，次要专家低比特量化，不重要专家删除或合并。

这比单纯剪枝更灵活。

---

# 9. 蒸馏：把 MoE 能力转移到小模型

Expert Distillation 的目标是把大 MoE 的能力转移到更小模型。

有两类：

## 第一类：MoE → 小 MoE

比如 LLaVA-MoD，把大多模态模型的能力蒸馏到小 MoE 模型。

## 第二类：MoE → Dense

比如 OneS、Switch Transformer、ELSM 等，把稀疏 MoE 蒸馏成 dense student model。

为什么这么做？

因为 dense model 虽然参数少一些，但部署更简单：

* 没有动态路由；
* 没有专家调度；
* 没有 all-to-all；
* latency 更稳定。

所以 sparse-to-dense 是一个重要方向。

---

# 10. Expert Decomposition：低秩分解专家

这类方法不是删专家，而是把专家权重矩阵分解成低秩矩阵。

例如一个大矩阵 (W) 分解为：

[
W \approx U \Sigma V
]

这样参数量和计算量都可以下降。

代表方法包括：

* MPOE；
* MC-SMoE；
* MoE-I2。

这类方法的关键思想是：

> 不同专家内部也存在冗余，可以用低秩结构压缩。

对你的课题来说，可以和剪枝结合：

先判断专家重要性；
重要专家用较高 rank；
不重要专家用低 rank 或直接剪掉。

---

# 11. 算法改进：动态 gating 和 sparse-to-dense

论文还讲了两类算法改进。

## 11.1 Dynamic Gating

传统 MoE 通常固定 top-k，比如每个 token 都选 top-2 专家。
Dynamic gating 认为：

> 不是每个 token 都需要同样多专家。

简单 token 可以只用 1 个专家；
复杂 token 可以用 2 个甚至更多专家。

代表方法包括：

* Li et al. Adaptive Gating；
* DynMoE；
* XMoE；
* AdapMoE；
* DA-MoE。

这类方法对推理优化很重要，因为它直接减少 active experts。

对剪枝研究也有启发：

> 专家剪枝不一定只发生在模型压缩阶段，也可以在推理时动态跳过专家。

也就是说，可以研究：

* 静态专家剪枝；
* 动态专家跳过；
* 静态 + 动态结合。

---

## 11.2 Sparse-to-Dense

这类方法把 MoE 模型转换成 dense 模型。
方法通常包括专家合并、知识蒸馏、结构转换等。

代表方法：

* XFT；
* OneS；
* TSEP；
* EWA；
* AdaMoLE。

这说明一个现象：

> 有时 MoE 的训练优势和推理优势不一致。MoE 训练/扩展很强，但推理部署可能更希望变成 dense 或更规则的模型。

---

# 12. 系统层优化：MoE 推理真正的工程难点

系统层主要包括两类：

1. Expert Parallelism；
2. Expert Offloading。

---

## 12.1 Expert Parallelism：专家并行

当模型太大时，一个 GPU 放不下所有专家，就把不同专家放到不同 GPU 上。

例如：

* GPU 0 放 expert 0、1；
* GPU 1 放 expert 2、3；
* GPU 2 放 expert 4、5；
* GPU 3 放 expert 6、7。

推理时，每个 token 被 router 选中专家后，需要被发送到对应 GPU。
这就产生了 all-to-all communication。

MoE 并行的典型流程：

1. 每个 GPU 先算 attention 和 gate；
2. 根据 gate 结果，把 token 发给对应 expert 所在 GPU；
3. expert 计算；
4. 把 expert 输出发回原来的 GPU；
5. 继续后续层。

这里的主要瓶颈是：

* 负载不均衡；
* all-to-all 通信；
* expert placement；
* task scheduling。

---

### 负载不均衡

如果大量 token 都选择 expert 3，而 expert 3 在 GPU 1 上，那么 GPU 1 很忙，其他 GPU 空闲，整体推理被拖慢。

解决方法包括：

* 给 router 加 load balancing loss；
* 复制热门专家；
* 动态调整专家放置；
* expert choice routing；
* 根据历史路由统计做调度。

---

### all-to-all 通信优化

MoE 多 GPU 部署最核心的系统瓶颈就是 all-to-all。

论文提到很多优化方式：

* hierarchical all-to-all；
* 减少通信数据量；
* 合并通信；
* token/expert placement 优化；
* 通信和计算重叠；
* 把 token 移动改成 expert 移动。

代表系统包括：

* Tutel；
* DeepSpeed-MoE；
* HetuMoE；
* Lina；
* ExFlow；
* Aurora；
* Janus；
* ScheMoE；
* PipeMoE。

这部分对于 AI infra 方向非常重要。

---

## 12.2 Expert Offloading：专家卸载

如果是在单 GPU 或边缘设备上，显存放不下所有专家，就需要把一部分专家放到 CPU 内存或 SSD。

推理时，只把需要的专家加载到 GPU。

这叫 expert offloading。

问题是：
如果每一层都等专家从 CPU/SSD 加载到 GPU，延迟会非常大。

所以系统需要：

1. **Expert Prefetching**：提前预测后面要用哪些专家；
2. **Expert Caching**：把常用专家留在 GPU；
3. **Expert Loading Optimization**：减少专家加载开销；
4. **CPU Assisting**：让 CPU 参与计算。

---

### Expert Prefetching

很多方法发现，相邻层的 gating 输入很相似，因此可以根据当前层预测下一层会用哪些专家。

代表方法：

* HOBBIT；
* Mixtral-Offloading；
* AdapMoE；
* Pre-gated MoE；
* EdgeMoE；
* ProMoE；
* ExpertFlow；
* Read-ME。

论文提到一些方法预测准确率能达到约 90%，因此预取非常有效。

---

### Expert Caching

因为 GPU cache 放不下所有专家，所以要决定哪些专家常驻 GPU。

常见策略：

* LRU：最近使用过的专家更可能再次使用；
* LFU：经常使用的专家更重要；
* 静态重要性 profiling；
* 动态 cache；
* cache-aware routing。

这对剪枝也有启发：

> 常驻 GPU 的专家通常是高频专家；低频专家可以考虑量化、卸载、合并或剪枝。

---

### Expert Loading

论文指出，在 HOBBIT 中，专家加载可能占总推理时间的 80% 以上。

这说明 MoE 推理的瓶颈很多时候不是算力，而是内存和数据搬运。

优化方法包括：

* 对低重要性专家使用低精度；
* cache miss 时动态选择加载高精度还是低精度专家；
* 跳过不重要专家；
* CPU 辅助计算。

---

# 13. 硬件层优化

论文最后总结了硬件层方法。

代表工作包括：

* MoNDE；
* FLAME；
* M3ViT；
* Edge-MoE；
* Duplex；
* Space-Mate。

这些方法的共同点是：

> 当前 GPU/CPU 主要为 dense computation 优化，而 MoE 是动态稀疏计算，所以需要专门硬件或软硬件协同设计。

例如：

## MoNDE

使用 near-data processing，把冷专家放到近数据计算单元里处理，减少专家参数搬运。

它的核心思想是从 **Parameter Movement** 转向 **Activation Movement**：

* 传统方式：把专家参数搬到计算单元；
* MoNDE：把 activation 搬到靠近专家参数的地方计算。

这对 MoE 很有意义，因为专家参数很大，activation 相对小。

---

## FLAME

面向 FPGA 的 MoE Transformer 加速框架。
利用 MoE 稀疏性和专家预测来提升效率。

---

## Duplex

结合 xPU 和 PIM，把不同层或不同阶段放到更合适的计算单元执行。

---

# 14. 论文指出的未来方向

论文认为 MoE 推理优化还存在很多开放问题。

主要包括：

## 14.1 硬件和系统软件需要重新设计

现有硬件和系统软件不太适合动态稀疏 MoE。

未来需要：

* 更好的 sparse computation 支持；
* 更好的 expert routing 硬件；
* 更好的 memory hierarchy；
* 更好的 expert prefetch；
* 更好的 MoE runtime。

---

## 14.2 能耗优化还不够

现有研究很多关注 latency 和 throughput，但较少关注 energy 和 carbon。

MoE 虽然计算稀疏，但通信、专家加载、调度可能带来额外能耗。

所以未来需要 energy-aware MoE inference。

---

## 14.3 QoS 和稳定延迟很难

MoE 的每个输入走不同专家，推理路径动态变化，因此 latency 不稳定。

生产环境中，不能只看平均 latency，还要看：

* p95 latency；
* p99 latency；
* cache miss 情况；
* 负载峰值；
* 热门专家拥塞。

---

## 14.4 缺少标准 benchmark

论文认为当前 MoE 推理优化缺少统一 benchmark。

不同论文用不同模型、不同硬件、不同 batch size、不同 baseline，很难公平比较。

未来需要更标准的 MoE benchmark，比如同时考虑：

* accuracy；
* latency；
* throughput；
* memory；
* communication；
* energy；
* cost。

---

# 15. 这篇论文对你的 MoE 剪枝课题有什么启发？

我觉得最重要的启发有 6 个。

---

## 启发 1：MoE 剪枝不能只看参数大小

普通 LLM 剪枝经常看 weight magnitude。
但 MoE 里专家的重要性还取决于 router。

所以剪枝指标应该考虑：

[
Importance(expert) = f(\text{weight}, \text{activation}, \text{routing frequency}, \text{routing score}, \text{task loss})
]

一个很少被激活的专家可能参数很大，但对当前任务没用；
一个参数范数不大的专家可能负责关键长尾能力。

---

## 启发 2：专家频率高不等于一定重要

很多方法用 expert activation frequency 判断重要性。
但这有问题：

* 高频专家可能只是通用专家；
* 低频专家可能负责罕见但重要能力；
* 某些 benchmark 不覆盖长尾能力；
* 剪掉低频专家可能伤害 out-of-distribution 能力。

所以你可以研究：

> 如何区分“真正冗余的低频专家”和“长尾关键专家”？

这是一个很好的研究点。

---

## 启发 3：删除专家和合并专家要结合

直接删除专家很简单，但可能损失大。
合并专家更温和，但实现复杂。

一个可能的研究路线是：

1. 先根据 router/activation 找出低重要性专家；
2. 判断它们和其他专家是否相似；
3. 如果相似，则 merge；
4. 如果不相似但贡献低，则 prune；
5. 如果是长尾专家，则保留但量化或卸载。

这比单一剪枝策略更合理。

---

## 启发 4：剪枝目标不应该只有 accuracy

MoE 剪枝的目标应该是多目标优化：

* accuracy；
* memory reduction；
* latency；
* throughput；
* expert load balance；
* communication cost；
* cache hit ratio；
* p99 latency。

比如剪掉某个专家虽然 accuracy 不掉，但会让剩下专家更拥挤，导致负载不均衡，推理反而变慢。

所以系统指标也要进入剪枝目标。

---

## 启发 5：分层剪枝很重要

不同层的专家冗余程度不同。

浅层可能处理通用特征；
中层可能负责语义模式；
深层可能更任务相关。

所以不能简单地每层剪同样比例。

更合理的是：

[
PruningRatio_l = f(\text{layer sensitivity}, \text{expert redundancy}, \text{routing entropy})
]

也就是每层自适应剪枝比例。

---

## 启发 6：剪枝可以和 offloading / quantization 联合

不是所有不重要专家都必须删掉。

可以设计多级策略：

| 专家类型      | 处理方式        |
| --------- | ----------- |
| 高频重要专家    | 保留在 GPU，高精度 |
| 中等重要专家    | 保留或低比特量化    |
| 低频但长尾重要专家 | 卸载到 CPU/SSD |
| 低频且冗余专家   | 合并或删除       |
| 与其他专家高度相似 | 合并          |

这会比单纯 prune 更适合真实部署。

---

# 16. 如果你要读这篇论文，建议这样读

这篇论文信息量很大，不建议从头到尾硬读。

你可以按你的课题这样读：

## 第一遍：只看整体框架

读：

* Abstract；
* Introduction；
* Figure 1；
* Section 2；
* Section 3 开头；
* Conclusion。

目标是理解 MoE 推理优化分哪几类。

---

## 第二遍：重点读剪枝相关

重点读：

* Section 3.2；
* Section 3.2.1 Expert Pruning；
* Table 2；
* Figure 4(a)。

这一部分与你课题直接相关。

---

## 第三遍：读和剪枝相关的系统因素

读：

* Section 4.1 Expert Parallelism；
* Section 4.2 Expert Offloading；
* Table 8；
* Table 9。

因为 MoE 剪枝最终服务于推理优化，不能只看模型压缩，还要知道系统瓶颈在哪里。

---

## 第四遍：读未来方向

读：

* Section 6。

这里可以帮你找 research gap。

---

# 17. 我帮你提炼几个可能的研究问题

结合这篇综述和你的课题，下面几个方向比较值得考虑：

## 方向 1：Router-aware Expert Pruning

利用 router logits、routing frequency、routing entropy、expert contribution 来判断专家重要性。

核心问题：

> router 信号能否可靠反映专家重要性？

---

## 方向 2：Task-agnostic MoE Pruning

不针对某个下游任务，而是保持通用能力。

难点：

> 如何避免剪掉低频但关键的长尾专家？

---

## 方向 3：Expert Similarity-based Merge-and-Prune

先衡量专家相似性，再决定合并还是删除。

关键问题：

> 专家相似性应该在权重空间、激活空间、输出空间还是路由空间衡量？

---

## 方向 4：Layer-wise Adaptive Expert Pruning

每层剪枝比例不同。

关键问题：

> 如何度量每层对剪枝的敏感性？

---

## 方向 5：System-aware MoE Pruning

把系统指标引入剪枝目标，比如 latency、communication、cache miss、load balance。

核心问题：

> 剪哪些专家不仅要看 accuracy，还要看是否真的减少推理瓶颈。

---

## 方向 6：Pruning + Quantization + Offloading 联合优化

不是简单删除专家，而是给不同专家分配不同处理方式。

可以建模成：

[
\min \quad \text{Latency} + \lambda_1 \text{Memory} + \lambda_2 \text{AccuracyLoss}
]

决策变量包括：

* 保留；
* 删除；
* 合并；
* INT8；
* INT4；
* CPU offload；
* GPU cache。

这个方向更偏系统 + 模型联合优化，非常适合 AI infra 推理优化方向。

---

# 18. 最后总结

这篇论文的价值在于：它把 MoE 推理优化从“单点算法”提升到了“全栈视角”。

它告诉我们：

> MoE 推理优化不是只减少 FLOPs，而是要同时考虑模型结构、专家路由、显存、通信、缓存、预取、硬件和系统调度。

对你的 MoE 剪枝课题来说，最重要的部分是：

* Expert Pruning；
* Expert Merging；
* Dynamic Gating；
* Expert Quantization；
* Expert Offloading；
* System-aware optimization。

如果你要从这篇论文中提炼自己的研究方向，我建议重点围绕：

**router-aware + layer-wise + system-aware 的 MoE 专家剪枝**。

这个方向既贴合你的“剪枝”主题，又能体现 AI infra / 推理优化特色。
