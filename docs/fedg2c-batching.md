# FedG2C 预训练与微调加速

2026-11-08 起，加速实现由 `master` 正式维护，直接在主目录使用。
Cora CUDA 配置沿用原有文件名；其他数据集可在统一配置中设置相同加速参数。

批处理改动前的本地代码保存在 Git 提交 `50648da`；根节点 query 加速前为
`bd136dd`。预训练保留单视图 forward；下游节点任务的标量和批量路径均只
计算根节点 query。原组合器的全 query 接口仍用于预训练、图分类和数值对照。
模块 B 和共享检查点的参数名称、形状及加载接口保持兼容。

## 执行参数

| 配置位置 | Cora CUDA 配置值 | 作用 |
| --- | --- | --- |
| `stage1.training.physical_batch_size` | 32 | 一次在设备上计算的独立视图数 |
| `stage1.training.checkpoint_interval` | 20（默认） | 每多少个已完成联邦轮次保存权重和探针；始终保存最终轮 |
| `matrix.source_batch_size.node` | 1024 | 累计多少视图后才裁剪梯度、执行 AdamW 和 EMA |
| `stage2.target.microbatch_size` | 4 | 微调时一次计算的独立邻域数 |
| `stage2.target.batch_size` | 0 | 整个训练支持集一次更新，物理批次之间只累计梯度 |
| `stage2.target.eval_batch_size` | 8 | 验证/最终测试时一次计算的样本数 |
| `stage2.target.crop_cache_mb` | 128 | 节点邻域拓扑缓存的张量字节上限；0 关闭 |
| `stage2.target.precompute_topology` | true | 训练前预热所选训练支持集和验证集的固定邻域与根节点关系 |

这些批次大小可根据新主机显存调整，无须改学习率或实验预算。配置未指定
批次时保持物理批次 1、验证批次 1。缓存不保存特征副本、标签、模型输出
或梯度；现有完整图与分片缓存不计入这个 128 MiB 拓扑张量预算。

矩阵配置按目标数据集覆盖缓存：Cora/CiteSeer 128 MiB、PubMed 256 MiB、
Photo/Computers 1024 MiB、WikiCS 2048 MiB。预算按需占用，不预先分配。
通用矩阵的其他节点目标默认 2048 MiB，仍可在 `matrix.target_overrides` 中调整。
预热日志 `target_topology` 报告工作集张量字节数、是否能完整放入缓存、命中和
驱逐计数；Python 对象、完整特征图和邻接索引的内存另计。超过预算仍按原规则
计算，可能发生 LRU 抖动；此时根据日志增大 `crop_cache_mb`。关闭预热只取消
提前构建，首次访问仍按预算缓存；`crop_cache_mb: 0` 完全禁用缓存。

## 根节点 query 与静态关系

节点任务只监督裁剪邻域中的根节点。组合器保留邻域所有专家 token 的 key/value，
只投影根节点的 query：分数从 `[B,H,P*M,P*M]` 缩为 `[B,H,M,P*M]`。
当前组合器只有一层，softmax 按 key、head RMS 按 head、LayerNorm 按特征维
计算；两次超网络收缩不混合不同 query，因此删除未使用的邻居输出不会改变
根节点的数学表达式。输入适配器仍通过全部邻域专家和 K/V 接收梯度。
图分类需要池化所有节点，继续使用全 query 路径。

每个固定邻域缓存节点编号、原顺序的有向边（含重复边）、原边属性行号、局部
根节点及其 `[P,3]` 布尔关系行。两跳关系始终使用整个裁剪邻域中的中间节点，
等于原 `[P,P,3]` 关系张量的根节点行；不从全图计算，不修改可见性。
训练支持选择完成后才预热训练和验证邻域；测试根节点仅在最终测试时访问。
特征、标签和训练中的专家表示不加入拓扑缓存，输入适配器与私人适配仍逐次
前向/反向。此优化没有改变样本顺序、优化器更新边界或选模规则。

## 保留的训练规则

- 每个客户端每轮完整遍历 5 次，保持原根节点顺序、局部随机生成器及
  边/通道遮蔽的随机调用顺序；采样和拼批不使用模型的随机数流。
- 每视图先平均其预测节点与特征上的 SmoothL1，再对逻辑批次的视图等权
  平均。预测节点数不同的视图也等权，不按拼接后的节点总数求平均。
- 物理批次不跨逻辑更新边界；epoch 尾部不足 1024 个视图时仍独立更新，
  按实际视图数归约，然后执行原来的梯度裁剪、AdamW 和 EMA。
- GNN 在不相连的图上计算；组合器的 batch 维独立，每张图保持自己的
  关系偏置、head RMS 和两次超网络收缩。padding 不参与注意力、损失、
  图池化或逐图诊断统计。LayerNorm 仍按每个节点的特征维归一化。
- random-walk 编码始终由当前视图遮边后的可见边计算；不能复用完整图
  的位置编码，也不能缓存训练中变化的主分支/动量分支输出。
- 下游保持原数据划分、可见性、支持集、样本顺序、逻辑更新次数、验证
  频率、准确率选模、首次同分优先、早停和选模后一次测试。

诊断累计和最后 128 个参考表示留在设备上，每次逻辑更新或训练结束时再
读取。每个物理批次仍检查非有限 loss，梯度裁剪仍检查非有限梯度。
改变浮点求和顺序会带来舍入差异，因而不承诺训练结果逐位相同。当前正式
配置的 dropout 为 0；非零 dropout 的预训练要求物理批次 1，以保留原来的
随机数调用顺序。

## 实现位置与验证

- `fedg2c/batching.py`：图边索引偏移、padding 映射、逐图关系与 random walk。
- `fedg2c/composer.py::forward_batch`：独立图 batch 维及 padding 屏蔽。
- `fedg2c/composer.py::forward_roots`：节点任务根节点 query，保留全部邻域 K/V。
- `fedg2c/edge_model.py::forward_batch`：主/动量双分支与按视图的损失归约。
- `fedg2c/edge_client.py::pretrain`：物理批次累计至原逻辑更新边界。
- `fedg2c/target_model.py::forward_batch` 和 `target_train.py`：四种适配的
  批量训练/评估；冻结基座仍允许梯度传递至可训练输入适配器。
- `fedg2c/target_data.py::TargetDataCache`：共享的有容量限制的拓扑缓存；
  `precompute_topology` 仅预热训练支持集和验证集，各 split 的标签独立读取。
- `fedg2c/context.py::structural_relation_rows`：完整裁剪图上的根节点关系，
  避免重复计算和传输全节点两两关系。

`python -m pytest -q` 检查标量/批量的输出和梯度、变长视图、不同预测节点数、
孤立节点、重复/有向边、类别特征、两种预测器、四种组合器、逐图隔离、
采样序列、epoch 残批、EMA 更新边界、四种下游适配及完整 pipeline/检查点加载。
默认测试目录限制在 `tests/`，不会扫描 `runs/` 中保存的其他仓库快照。
新增对照覆盖四种组合器、四种适配（含非零 LoRA/Adapter/Prompt）、局部分支、
可学习混合系数，以及缓存启用/禁用、字节限制和测试集访问时机。
在 CUDA 主机上可运行 `FEDG2C_TEST_DEVICE=cuda:0 python -m pytest -q`。

## 新主机运行

正式运行配置为 `config/fedg2c_gefra_loo_v1_cora_cuda.json`。它使用
100 轮、5 本地 epochs、5 源客户端、模块 B 10 步；Cora 最多 1000 epochs，
每 10 epochs 验证、5 个未改善验证事件早停；full/5-shot/1-shot、10 个种子、
none/LoRA/Adapter/Prompt 共 120 组下游实验，选模后评估测试集。

在仓库根目录先检查执行计划，再运行只包含 Cora 的脚本：

```bash
python -m fedg2c.gefra_loo_matrix --config config/fedg2c_gefra_loo_v1_cora_cuda.json --targets cora --dry-run
bash scripts/fedg2c/run_cora_formal_gpu.sh
```

数据缓存仍由 `cache_data` 指向已有缓存，预处理结果使用
`runs/gefra_loo_v1_data/node/cora`。数据准备会复核协议、种子及源/目标角色。
输出写入新的 `runs/gefra_loo_v1_cora_batched/node/cora`，已有输出不会覆盖。
这是从统一初始化重新运行，不是恢复此前中止的优化器/私人适配器状态。

独立短基准不会改正式配置、检查点或启动完整 pipeline：

```bash
# 本地合成执行基准，不能作为 Cora/GPU 性能结果
python scripts/fedg2c/benchmark_batching.py --device cpu --views 64 --repeats 2

# 数据已准备好后，在新 GPU 主机测量实际 PubMed 源客户端
python scripts/fedg2c/benchmark_batching.py --device cuda:0 --source-manifest runs/gefra_loo_v1_data/node/cora/source_manifest.json --client pubmed --views 128 --repeats 3
```

基准的两个路径从相同参数和随机种子开始，均在指定视图数之后更新一次，
报告吞吐、loss 差异和 GPU 峰值分配显存。这个短预算仅用于性能测量，
正式实验仍由配置中的 1024 视图更新边界决定。GPU 加速比与完整一轮耗时
需要在新主机测量；CPU 合成基准不能用于外推正式实验耗时。

2026-11-02 本地检查：PyTorch 2.10.0+cpu、单线程、64 个合成视图、两次
测量的中位数，物理批次 1 为 0.743 秒（86 视图/秒），物理批次 32 为
0.140 秒（457 视图/秒），约 5.31 倍，报告 loss 差异为 0。CPU 测试共
46 项通过；CUDA 数值对照、显存峰值和实际数据吞吐待新主机验证。

根节点加速另有独立基准（专家编码、组合器、输入/私人适配器的前向与反向）：

```bash
python scripts/fedg2c/benchmark_target_queries.py --device cpu --batch-size 2 --nodes 128 --repeats 3
# GPU 上可使用正式邻域上限和物理批次；不会写检查点或启动正式训练
python scripts/fedg2c/benchmark_target_queries.py --device cuda:0 --batch-size 4 --nodes 256 --repeats 3
```

它从相同权重和输入比较旧全 query 与新根节点 query，报告输出/梯度最大误差、
用时及 CUDA 峰值显存。输入为合成图，实际数据完整 pipeline 的耗时需另测。

2026-11-04，本地 PyTorch 2.10.0+cpu、单线程、LoRA、2 个 128 节点邻域、
5 专家、128 维、3 次测量的中位数：全 query 0.214 秒、根节点 query
0.0138 秒（15.5 倍）；输出最大绝对差 0、参数梯度最大绝对差约 `3e-7`。
这不是 GPU 或真实数据吞吐结果，不用于承诺完整实验耗时。


2026-11-08 主线整合验证：在服务器 Python 3.11.11、PyTorch 2.6.0+cu124、
PyG 2.6.1 的 CPU 环境，完整测试 `python -m pytest -q` 共 79 项通过
（20.82 秒）。其中额外核对批处理与每 20 轮保存策略组合后的最终参数一致性、
统一配置自定义保存间隔以及四种适配的检查点重载。13 个 pipeline/矩阵配置
展开的 98 份计划通过配置解析，Cora CUDA 配置的 dry-run 正常。
本次主线整合时 CUDA 不可用，未重新测量 GPU 显存、吞吐或完整训练耗时。
