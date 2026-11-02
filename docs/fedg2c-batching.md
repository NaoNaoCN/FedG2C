# FedG2C 批处理与 Cora 重跑

当前改动前的本地代码保存在 Git 提交 `50648da`。加速实现保留单视图
forward 作为对照；设物理批次为 1 可使用该路径。模块 B 和共享检查点的
参数名称、形状及加载接口保持兼容。

## 执行参数

| 配置位置 | Cora CUDA 配置值 | 作用 |
| --- | --- | --- |
| `stage1.training.physical_batch_size` | 32 | 一次在设备上计算的独立视图数 |
| `matrix.source_batch_size.node` | 1024 | 累计多少视图后才裁剪梯度、执行 AdamW 和 EMA |
| `stage2.target.microbatch_size` | 4 | 微调时一次计算的独立邻域数 |
| `stage2.target.batch_size` | 0 | 整个训练支持集一次更新，物理批次之间只累计梯度 |
| `stage2.target.eval_batch_size` | 8 | 验证/最终测试时一次计算的样本数 |
| `stage2.target.crop_cache_mb` | 128 | 节点邻域拓扑缓存的张量字节上限；0 关闭 |

这些批次大小可根据新主机显存调整，无须改学习率或实验预算。配置未指定
批次时保持物理批次 1、验证批次 1。缓存不保存特征副本、标签、模型输出
或梯度；现有完整图与分片缓存不计入这个 128 MiB 拓扑张量预算。

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
- `fedg2c/edge_model.py::forward_batch`：主/动量双分支与按视图的损失归约。
- `fedg2c/edge_client.py::pretrain`：物理批次累计至原逻辑更新边界。
- `fedg2c/target_model.py::forward_batch` 和 `target_train.py`：四种适配的
  批量训练/评估；冻结基座仍允许梯度传递至可训练输入适配器。
- `fedg2c/target_data.py::TargetDataCache`：共享的有容量限制的拓扑缓存；
  各 split 的标签和少样本支持选择独立读取。

`python -m pytest -q` 检查标量/批量的输出和梯度、变长视图、不同预测节点数、
孤立节点、重复/有向边、类别特征、两种预测器、四种组合器、逐图隔离、
采样序列、epoch 残批、EMA 更新边界、四种下游适配及完整 pipeline/检查点加载。
默认测试目录限制在 `tests/`，不会扫描 `runs/` 中保存的其他仓库快照。
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
