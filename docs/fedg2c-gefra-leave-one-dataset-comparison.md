# FedG2C 与 Gefra 留一数据集迁移对照

本文只分析 Gefra 原生的跨数据集留一迁移协议。每次实验选择一个目标数据集，其余同类数据集各作为一个源客户端；PCBA→HIV fixed-source/few-shot 协议不属于本文范围。

## 1. Gefra 的实际协议

Gefra 的配置文件 `config/pretrain_config_<target>.json` 同时定义源池、客户端数和目标。例如四个图分类目标为：

| 目标 | 源客户端（每个数据集 1 个） | 专家数 |
|---|---|---:|
| `chemhiv` | `chempcba`, `imdb-binary`, `proteins` | 3 |
| `chempcba` | `chemhiv`, `imdb-binary`, `proteins` | 3 |
| `proteins` | `chemhiv`, `chempcba`, `imdb-binary` | 3 |
| `imdb-binary` | `chemhiv`, `chempcba`, `proteins` | 3 |

六个核心节点数据集同样使用留一规则。例如 `cora` 的源池为 `citeseer`, `pubmed`, `photo`, `computers`, `wikics`，专家数为 5；`pubmed` 的源池为 `citeseer`, `cora`, `photo`, `computers`, `wikics`，专家数也为 5。其它目标使用显式源池：Arxiv 来自 Photo/Computers/WikiCS，Reddit 来自 Cora/CiteSeer/PubMed，Airport/Disease 来自六个核心节点源。不能把所有已支持目标解释为同一个大池上的机械留一。`wn18rr`/`fb15k237` 是关系分类任务（不是二元有无边预测），FedG2C 当前新路径没有对应的目标数据入口。

Gefra 的完整原生图流程是：

1. `pretrain_gefra.py` 读取目标对应的 JSON；源数据集各建立一个客户端，目标数据集不进入源池。
2. 每轮服务器下发共享 MoE 专家和几何投影器；每个源客户端对自己的完整图集合运行 5 个本地 epoch，再上传共享参数、专家使用分布和专家关系低秩因子。
3. 服务器先用专家关系矩阵计算 Gromov-Wasserstein 聚合权重，再用 Hungarian 匹配对齐客户端专家索引，最后对专家和几何投影器做加权平均。默认 100 个通信轮次；服务器没有 FedG2C 那种可学习混合 logits 的内部优化步。
4. `finetune_gefra.py` 只加载目标客户端和最后的服务器检查点；目标阶段不再联邦通信。目标训练集更新输入投影、本地编码器、Prompt Readout、任务头等下游参数；默认冻结共享 MoE/几何投影器。
5. 节点/图分类目标默认分别运行 `k_shot=-1, 5, 1`；关系目标只默认 `-1`。Shell launcher 对节点/图任务设置 `standard=10`，直接 Python CLI 的默认值则为 1。`standard` 循环内部使用 seed `0..standard-1`，共享同一个源检查点、划分和 support；目标客户端先初始化，再在循环里 deepcopy，因此也不能称为十套完全独立的目标初始化。每个目标的 epoch、早停、batch size 和学习率来自 `config/finetune.yaml`。

### 数据划分的关键事实

Gefra 的 `load_fedgfm_datasets` 先为 OGB 数据读取官方 scaffold 索引，但 `graph_set_partition` 随后在每个客户端分片内重新生成确定性的随机 `0.8/0.1/0.1` 图级 train/valid/test mask。图预训练使用 `*_pt.pt`，遍历客户端的全部图并不读取标签；目标微调使用 `*_ft.pt`，只用 train mask 更新，valid/test 用于评估。因此，原生 Gefra 图迁移并不是当前 FedG2C M1 的“官方 scaffold 子集 + 4096/1024/1024”协议。

节点任务则使用 Gefra 的固定比例：Cora/CiteSeer/Photo/Computers 为 `20/40/40`，PubMed 为 `60/20/20`，WikiCS/Arxiv/Reddit/Disease/Airport 为 `80/10/10`；按类别确定性抽样。关系目标使用另一套边级划分。

### Gefra 的重要默认值

| 阶段 | 参数 | Gefra 设置 |
|---|---|---|
| 源预训练 | 通信轮次 | 100 |
| 源预训练 | 每轮本地训练 | 5 个完整 local epoch |
| 源预训练 | 图任务 batch | 64（`pretrain_batch_size=1024` 只用于节点/关系邻居采样） |
| 源预训练 | AdamW lr / weight decay | `1e-3 / 1e-5` |
| 源预训练 | RiemannianAdam lr | `1e-3`，几何参数单独优化 |
| 源预训练 | 梯度裁剪 | 全部参数 `max_norm=1.0` |
| 源预训练 | 特征遮蔽 / 边丢弃 | `0.2 / 0.2` |
| 模型 | hidden dim / 层数 | `128 / 2` |
| 模型 | Top-K | 2；MoE 专家数量按目标源池配置 |
| 模型 | backbone / norm / dropout | 原生共享专家实际固定为两层 `KappaGCN`；CLI `gnn-backbone=gcn` 作用于目标/客户端 local encoder 与欧式专家消融。欧式路径用 BatchNorm/dropout 0.15；不能把这些值认作原生几何专家的实际结构 |
| 下游 | ChemHIV | 400 epochs, early stop 40, batch 1024, lr `5e-4` |
| 下游 | ChemPCBA | 30 epochs, early stop 50, batch 512, lr `1e-4` |
| 下游 | PROTEINS | 200 epochs, early stop 50, batch 128, lr `1e-4` |
| 下游 | IMDB-BINARY | 200 epochs, early stop 50, batch 512, lr `1e-4` |
| 下游 | 评估间隔 | 每 10 个 epoch；早停 patience 按评估事件折算 |
| 下游 | 选模指标 | 原生 Logger 以验证 ROC-AUC 最大化；同时记录测试 ROC-AUC/AP |
| 下游 | AdamW weight decay | 代码未传参，使用 PyTorch 默认 `0.01`；FedG2C 当前为 `0.0` |

Gefra 图预训练的损失不是一个简单的分类损失：`topo_recon_loss`、`struct_align_loss` 和 `curvature_div_loss` 的权重分别为 `0.1`、`5` 和 `0.3`。它对输入特征做 `mask_feature(p=0.2)`，对边做 `dropout_adj(p=0.2)`，用拓扑重建、局部/共享结构对齐和曲率分离进行自监督学习。

## 2. FedG2C 当前流程

FedG2C 的统一入口 `run_fedg2c_pipeline.py` 将阶段一和阶段二写入一个 JSON：

1. 阶段一由 `pretrain_federated_fedg2c.py` 依次向每个源客户端下发 `experts.*` 与 `composer.*`，本地 `EdgePredictionClient` 从有界邻域采样视图，遮边和特征通道，用动量分支产生逐节点参考表示，GraphSAGE 预测器优化 Smooth-L1 联合嵌入预测。
2. 客户端只上传共享专家和完整组合器；私有输入适配器、预测器和动量分支不上传。双方都保留私有输入模块，但共享对象不同：Gefra 只共享专家和几何投影器，其预训练 gating 不共享/不下游迁移；FedG2C 必须共享和迁移完整 composer。这是方法差异，不应为了“参数一致”删掉 composer。
3. 阶段一每轮服务器执行 FedAvg、output-only 或 response 聚合。当前 response 模式从 SBM 合成探针构造输出/有限差分响应目标，在客户端混合 logits 上用 Adam 优化若干步；它不是 Gefra 的 GW/Hungarian 聚合复刻。
4. 阶段二 `target_train.py` 重建目标域输入适配器和任务头，默认冻结共享基座，支持 `none/lora/adapter/prompt` 四种独立适配。当前统一配置可以用最低验证 BCE 选最佳检查点，并在选定后才测试。

当前已完成的分子配置是 M1：PCBA 取 10,000 张源图，PROTEINS/IMDB 使用缓存集合，MolHIV 取 `4096/1024/1024` 图；使用 OGB scaffold train/valid/test、hidden dim 64、4 专家、`mysage`、源每轮 1,000 个优化步。这是工程子集协议，与 Gefra 全量图集和随机 80/10/10 图划分不同。

## 3. 可以直接沿用的 Gefra 设置

这些是实验协议层面的设置，不改变 FedG2C 的学习目标：

| 可沿用项 | 在 FedG2C 中的落实方式 |
|---|---|
| 留一数据集角色 | 为每个目标生成一个配置；目标从源客户端清单中排除 |
| 每个源数据集一个客户端 | 继续保持一数据集一客户端，不把一个数据集再切成多个 FedG2C 客户端 |
| 源/目标客户端顺序和稳定 ID | 写入 manifest，固定客户端顺序、数据哈希和协议 ID |
| 100 轮、每轮 5 个本地 epoch | 可采用这个协议目标；须先落实 graph batch/epoch 语义，当前单图一步实现不能直接达到可比计算预算 |
| AdamW `1e-3/1e-5` | FedG2C A 的默认值已相同；双方每轮重建欧式 AdamW。Gefra 几何优化器持续存在，FedG2C 没有该参数组 |
| hidden dim 128、2 层 | 作为可比配置；需同步更新适配器、组合器和 checkpoint schema |
| 特征遮蔽率 0.2 | 可作为 FedG2C 的 feature mask 起点，但仍使用 FedG2C 的类别安全遮蔽实现 |
| 目标阶段冻结共享基座、无下游联邦通信 | 当前 FedG2C 已满足，并通过基座哈希检查 |
| 目标数据的 train/valid/test 身份和 seed 固定 | 可沿用；每个目标的 train/valid/test 必须生成一次并跨适配方式复用 |
| 评估同时报告 ROC-AUC 与 AP | 当前 FedG2C 已报告；不把 AP 隐藏在单一 BCE 数字后面 |

## 4. 不能直接照搬的设置

| Gefra 设置 | 不能直接复制的原因 | FedG2C 应保留 |
|---|---|---|
| GW barycenter + Hungarian + 几何投影器 | 这是 Gefra 的核心几何聚合；替换 response server 会改变 FedG2C 模块 B 的研究问题 | SBM 探针、响应有限差分和 response/output/prior 目标 |
| Riemannian MoE、曲率分离损失、RiemannianAdam | FedG2C 明确是 geometry-free 的专家组合；引入曲率参数会把方法改回 Gefra | 欧式专家、关系条件组合器、AdamW |
| Gefra 的拓扑重建/结构对齐预训练目标 | 会替换 FedG2C 的逐节点联合嵌入预测与动量参考机制 | `edge_mask_rate`、`feature_mask_rate` 仅作为 FedG2C 自身增强参数重新校准 |
| `topk=2` 路由 | FedG2C 组合器对所有专家 token 做关系条件组合，没有同构的 Top-K 路由变量 | 保留 FedG2C composer 和 `num_experts`，另把专家数作为容量对照因素报告 |
| Gefra 的 `dropout_adj(p=0.2)` | FedG2C 要保证无向边按无序边组一次采样，且预测邻域只从可见图构造；直接替换会改变 A 的信息泄漏边界 | 保留 FedG2C 的成对遮边实现；可把 rate 设为 0.2 做协议敏感性对照 |
| Gefra 原生 graph feature path | Gefra 将 OGB/TU 特征转 float 后线性投影；FedG2C 使用 OGB 类别 embedding、边属性 embedding 和私有 schema | 保留 FedG2C 的私有输入适配器；若追求输入完全一致，应另做显式 `gefra_compat` 特征模式并单独标识 |
| Gefra 的目标分类头和 Prompt Readout | FedG2C 的 `none/lora/adapter/prompt` 位置、参数和作用域不同 | 保留 FedG2C 四种适配的互斥实验定义，不宣称它们等价于 Gefra Prompt Readout |
| Gefra 的目标 local encoder | Gefra 目标客户端重新随机初始化 local encoder，并训练它与共享分支的融合权重；FedG2C `local_alpha=0` 默认不创建/不训练该分支，所以 FedG2C 的 `none` 不是 Gefra 默认下游的同义词 | 先保留 `local_alpha=0` 作为 FedG2C 原生条件；另设 `local_alpha>0` 的明确对照，不能把两者合并为一个结果 |
| Gefra 每 10 epoch 读取测试集 | 这会让调参过程接触测试预测，且与当前已确认的 validation-only 规则冲突 | 保留“验证选模，最终一次测试”的 FedG2C 规则；若复刻 Gefra，只能作为单独 protocol 标记 |
| 旧的 M1 数据文件和历史分数 | Gefra 用全量图集与随机 80/10/10；同一数据集名称不代表相同实验身份 | 可在新协议中复用 Gefra 索引，但不能直接复用 M1 分数/manifest 冒充对齐结果 |

## 5. 对齐后仍需明确记录的差异

“可比”不等于“实现相同”。至少要在报告中同时列出以下差异：

1. Gefra 的源图预训练遍历完整源数据集；FedG2C 若采用 100×5 个完整 epoch，计算预算会随数据集大小自然变化。当前 FedG2C 的 1,000 steps/轮是等步数协议，不能和它混用。
2. Gefra 的图数据通常使用原始 OGB/TU 特征和完整图；FedG2C 的有界 32 节点视图、随机游走编码和边属性适配是方法本身的一部分。
3. Gefra 的图目标是原生随机 80/10/10 mask；FedG2C 当前目标 loader 是显式 `split_graphs` 文件。两者可以保存相同索引，但不能只改 manifest 名称而假装一致。
4. Gefra 的 `num_experts` 按源数据集数量配置（分子图留一为 3，节点图留一通常为 5 或 6）；FedG2C 目前分子配置固定为 4。做方法对照时建议提供“Gefra-matched expert count”和“FedG2C-fixed expert count”两列，而不是把容量差异归因于聚合方法。
5. Gefra 的下游图分类以验证 ROC-AUC 最大化；FedG2C 当前已确认的预算实验以最低验证 BCE 最小化。主结果应选择一种预注册指标，另一种只作伴随报告；不能在同一矩阵里按不同指标挑模型。

### 当前代码不能仅靠配置解决的缺口

- **源训练 batch。** Gefra 每 64 张图一次更新，FedG2C 每张图生成一个局部视图后立即更新。当前 `local_budget.unit=epochs, amount=5` 会做 `5N` 次更新，Gefra 为 `5 ceil(N/64)` 次。按缓存 PCBA 的 437,929 图计算，100×5 的两个预算分别约为 2.19 亿和 342 万次更新，差约 64 倍（尚未扣除无法采样的图）。应增加每 64 个独立视图汇总一次梯度的实现，明确 loss 的按视图/预测节点归约；不要把多图 token 直接拼接后让 composer 跨图注意。
- **节点 epoch。** 当前节点导出中，每个客户端只有一张大图；一次图遍历仅随机选一个根邻域，不是 Gefra NeighborLoader 对全部 seed node 的一遍。沿用 `5 epoch` 前，需要定义节点根的完整遍历和 batch；不能把五个随机邻域称作五次全节点遍历。
- **四个目标与 few-shot。** `prepare_molecular_data.py` 当前固定 PCBA→HIV M1，`prepare_node_data.py` 固定五源→Cora；没有通用目标轮换和 `k_shot` 配置。要导出所有目标并保留原始索引、mask、support 身份。Gefra 的图 few-shot 从 train IDs 的固定顺序取每类前 k 个，PCBA 对 128 个 task/class 所取 ID 做并集，因此 `5-shot` 的实际图数不是 5，也不一定是 10；如复刻，应直接保存同一 support ID，而非换一个随机采样器。
- **完整图大小与显存。** 远端缓存 MolPCBA 共 437,929 图、最大 332 节点；MolHIV 共 41,127 图、最大 222 节点。当前目标 `max_nodes=256` 无法保证覆盖全部 PCBA/TU 图。FedG2C composer 的计算随 `(节点数×专家数)^2` 增长；应先测目标全图规模、采用保持图内计算的流式加载/微批/梯度累积。不能丢大图或裁图后与全量 Gefra 分数比较。现有源采样要求有非自环边，对无可用边的源图需记录固定 skip 规则或双方统一过滤集合，不能悄悄少算样本。
- **验证间隔。** 当前 `stage2.target` 没有 `eval_interval` 或多 repeat 字段；每轮验证一次。Gefra 是零基 epoch `0,10,20,...` 及最后一轮，`early_stop=40` 表示四次无改善的验证事件；直接将 FedG2C patience 改为 40 不等价。
- **输入信息。** Gefra IMDB 用 `OneHotDegree(max_degree=135)`，FedG2C 用每图最大度归一化的单列 degree。Gefra OGB 是 9 列类别编码转 float 后线性投影；其原生 `KappaGCN` 和 `GraphEncoder.encode` 实际不使用 bond 属性。FedG2C 使用类别 embedding 和 bond 属性。可以统一原始输入/显式关闭边属性作受控版本，也可以保留原生处理并明确承认输入信息差异；不能把差异全归因于 A/B。
- **归一化/正则。** FedG2C 专家用 LayerNorm，Gefra 欧式 local encoder 用 BatchNorm；直接换 BN 会产生客户端运行统计量和共享 buffer 如何处理的问题。`dropout`、clip `1/5`、下游 weight decay `0.01/0` 可以作为对照参数，但不要把归一化替换视作无影响的配置复制。

### 指标、选模与统计口径

图任务的公共主指标可以沿用 Gefra 的 macro ROC-AUC（逐任务计算后对有效 task 取平均）；两边的监督二元损失都是按有效标签归约的 BCE。MolPCBA 必须保留 128 个任务、NaN 缺失标签和任务数，不能退成单标签分类。Gefra 的 AUC 输出为百分数，而 FedG2C 为 `[0,1]`，汇总时必须统一单位。Gefra 附加 AP 对多任务矩阵 flatten，FedG2C 则对各 task 的 AP 做 macro 平均，PCBA 上这两个 AP 数不能直接对比；应统一额外的 macro-AP 定义并报告有效任务数。对单任务 HIV 两者没有这一 flatten/macro 差异。

还有两处实现细节不宜盲目复制：Gefra `Logger` 验证分数并列时保留较晚 epoch，早停却只把严格改善视为改善；FedG2C 保留最早并列。Gefra 无有效 AUC task 时返回 0.5，FedG2C 返回 None 并拒绝无定义选模指标。新对照建议明确采用严格改善、首次并列和未定义指标报错，避免把默认 0.5 当作实测结果。

Gefra `standard=10` 的含义是条件于固定源检查点/数据划分/support 的十次微调随机过程，不是十次独立联邦预训练。若要同时检验 FedG2C 模块 A/B 的稳定性，应另有多预训练 seed 的整流程实验，不能混成相同误差条。

## 6. 推荐的新对照协议

在不改变 FedG2C 方法的前提下，建议新增 `gefra_loo_v1`，不重跑或覆盖 `molecular_m1`：

- 目标先覆盖四个图任务：`chemhiv`、`chempcba`、`proteins`、`imdb-binary`；每个目标使用其余三个完整数据集作为源客户端。
- 数据使用 Gefra 同一缓存和随机 80/10/10 图级划分；源预训练不导出标签，目标 train/valid/test 只生成一次并由所有适配方式复用。
- 主对照建议先用 `hidden_dim=128, num_experts=3, num_layers=2`；它们不会把欧式专家和关系条件 composer 改成几何模型。保留 4 专家的旧配置作为单独容量对照，不预先扩大为完整额外实验矩阵。
- 阶段一的目标预算为 `rounds=100`、完整客户端遍历 `5 epoch/round`，每 64 个独立源图视图一次更新；应先完成 batch 实现和预算核验，再启用此配置。保留有界局部视图、预测器、EMA、relation-conditioned composer。模块 B 保持 response 模式和原始目标函数；服务器 10/20 步是方法专属预算，没有对应的 Gefra 步数。
- 阶段二可以沿用 Gefra 的目标学习率/batch/epoch 上限和验证间隔，但保留 FedG2C 的冻结基座与四种适配。新对照的选模指标仍需用户确认，不擅自改掉当前最低 BCE 规则。先做固定源 checkpoint 下与 Gefra `standard=10` 匹配的微调重复；整流程多源 seed 是另一项研究计划。
- 下游先固定 `local_alpha=0`，把它标为 FedG2C-native；若要声明“下游流程也与 Gefra 对齐”，再单独运行 `local_alpha>0`（目标 local encoder 随机初始化、只在目标训练中更新）的 matched 条件。

该协议可以回答“FedG2C 在 Gefra 留一数据角色和数据规模下是否仍工作”，不能回答“FedG2C 是否复现 Gefra 的几何聚合算法”。后一问题需要另做 Gefra 原生基线，而不应把 Gefra 聚合搬进 FedG2C。

留一数据集迁移也不自动构成“未见组合”验证：原生协议没有显式留出关系/组合因素。它适合做通用跨域迁移对照；FedG2C 原本的组合泛化主张仍需另外的受控组合留出与模块 B 消融证据，不能仅用 LODO AUC 改善替代。

新实验开始前只需先确定研究协议，而不是修改方法：第一，直接复用 Gefra 的随机 80/10/10，还是双方重跑同一份官方 scaffold（后者不能直接拿 Gefra 旧分数）；第二，维持当前 BCE 选模并让 Gefra 也遵守，还是仅对新对照改成 Gefra 的验证 ROC-AUC；第三，使用相同输入特征信息的受控版本，还是保留各自原生输入并明确列出这一差异。这三项在本次分析中均未执行。

## 7. 代码对照索引

| 责任 | Gefra | FedG2C |
|---|---|---|
| 目标选择/源池 | `config/pretrain_config_<target>.json`, `pretrain_gefra.py` | `data.source_manifest`, `data.target_manifest`, `fedg2c/pipeline.py` |
| 图数据划分 | `utils/partition_utils.py:395-488` | `fedg2c/prepare_molecular_data.py`, `fedg2c/target_data.py` |
| 本地预训练 | `flcore/client.py:225-352` | `fedg2c/edge_client.py`, `fedg2c/edge_model.py`, `fedg2c/edge_views.py` |
| 联邦轮次 | `pretrain_gefra.py:218-230` | `pretrain_federated_fedg2c.py` |
| 服务器聚合 | `flcore/server.py:125-198,343-416` | `fedg2c/response_server.py` |
| 下游训练/选模 | `flcore/client.py:467-607`, `task/graph_cls.py` | `fedg2c/target_train.py`, `fedg2c/target_model.py` |
| 下游参数 | `config/finetune.yaml` | `stage2.target` in the unified JSON |

本文依据的 Gefra 代码快照来自 `chenyh26@192.168.5.92:32100:/home/chenyh26/FedGFM2oE/Gefra`，commit `71856218e19d7da72cf443825aa5ad0131e617f3`；快照和哈希记录保存在 `runs/gefra_setting_audit_20261029/`。
