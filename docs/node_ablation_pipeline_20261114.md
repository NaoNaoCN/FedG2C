# 节点分类三组消融：代码审阅记录（2026-11-14）

远端 `chenyh26@192.168.5.97:31840`。独立工作区 `/home/chenyh26/FedG2C_worktrees/ablation_pipeline_20261114`，分支 `codex/ablation-pipeline-20261114`，基于主线 `5698ac4`。修改尚未提交或推送，未同步本地源代码；正式实验未启动。

## 已确认的实验定义

| 组 | 模块 A | 模块 B | 下游 LoRA 注入位置 |
| --- | --- | --- | --- |
| baseline | 原关系条件超网络组合器 | 原关系响应保持聚合 | composer.query、composer.key |
| ablate_b | 原关系条件超网络组合器 | 源客户端样本量加权 | composer.query、composer.key |
| ablate_ab | 现有纯 linear_gate | 源客户端样本量加权 | composer.gate.0、composer.gate.2 |

不设置只消融 A：纯门控不读取关系门控，原 B 对关系门控的有限差分响应会恒为零，无法完整保留原 B 的约束。baseline 与 ablate_b 比较 B 在原 A 下的作用；ablate_b 与 ablate_ab 比较 A 在样本量加权聚合下的作用。这是三个设置的比较，不是完整的 2×2 因子实验。

目标仅为 Cora、CiteSeer、PubMed、Photo、Computers、WikiCS；各目标用其余五个节点数据集作源客户端。共准备 18 个预训练任务、540 个下游 LoRA 任务。每个目标和设置一个预训练 seed=2026；下游 full/5-shot/1-shot、seeds 0–9，沿用确定性固定支持集。

三组均需重跑，包括新 baseline。用户提供的 RESULTS.md 对应只有方差正则、没有本次输入偏置归零的历史结果，保留作参考，不混作新初始化协议下的 baseline。

## 正式配置中的两项改进

`config/fedg2c_nodes_ablation_20261114.json` 从历史方差实验冻结配置派生：

```json
{
  "variance_weight": 1.0,
  "variance_target": 0.15,
  "variance_eps": 0.0001,
  "variance_representation": "both",
  "variance_view": "reference",
  "private_adapter_bias_init": "zero"
}
```

方差项是组合输出与专家输出的标准差下限之和，统计轴是同一完整逻辑 batch 的源根节点；专家输出按每个专家和通道独立统计。使用在线分支在未增强 reference 视图上的表示，EMA 分支不接收梯度。每个标准差下限为 `mean(relu(0.15 - sqrt(var(roots, unbiased=False) + 1e-4)))`。总损失为预测损失加该方差项，系数为 1。

延用已经验证的双遍方法：先以物理分块收集完整逻辑 batch 的根表示，再将完整统计量的梯度逐块回放。物理 batch 不定义方差统计，也不改变优化器和 EMA 的更新边界。启用时要求 node 采样、每根节点一个 view、dropout=0；单样本尾批的方差项为零。

输入偏置仅在构造源端私有连续输入适配器时原位置零，保留 PyTorch 原随机抽样顺序，偏置仍可训练；EMA 从处理后的在线分支复制。每轮加载共享参数时不重新归零私有参数。专家、组合器及下游输入适配器的原初始化策略沿用。类别嵌入没有该线性偏置。

`EdgePredictionConfig` 对缺失字段保留历史默认值（方差权重 0、偏置策略 default），避免旧配置或旧 checkpoint 被错误解释；新的正式节点配置显式打开两项改进。

## 样本量加权聚合

新增独立模式 `sample_weighted`，不改变历史 `fedavg` 固定权重模式和 `response` 的原权重学习算法。

`w_i = n_i / sum_j(n_j)`，共享参数按该权重直接凸组合。此模式不采样探针、不优化服务器权重；报告记录客户端顺序、原始计数、归一化权重及 optimization_steps=0。非参数缓冲区仍要求各客户端相同。

对于节点任务，`n_i` 是允许源图中有非自环关联边的不同根节点数，与训练采样器的 eligible_roots 完全一致；孤立节点和只有自环的节点不计入。对于图任务为源图记录数。计数不乘 epoch、view 或优化步数，也不使用下游标签和支持集样本数。

已核对源数据：Cora=2708、CiteSeer=3279、PubMed=19717、Photo=7535、Computers=13471、WikiCS=11364。每个目标从这六个计数中排除自身，并按实际 manifest 的客户端顺序归一化。

联邦入口上传真实计数并写入客户端 checkpoint 元数据。独立聚合入口可从 checkpoint 读取；对于缺少计数的旧 checkpoint，必须显式提供 `--client-sample-counts`，不静默退回均匀聚合。

## LoRA 和保留的微调协议

所有设置 LoRA rank=4、alpha=4；原组合器注入 Q/K，纯门控注入两层 Linear。原共享基座冻结，低秩权重不与基座合并。报告新增实际 LoRA 注入路径和参数量。

hidden_dim=128 时，Q/K LoRA 共 2048 个参数，门控 LoRA 共 1540 个参数。本次不强行匹配参数量。按用户确认，保留原微调初始化顺序：相同 seed 编号和相同支持集不意味着不同结构的目标适配器与分类头初值逐项相同。`fedg2c/target_train.py` 未修改。

预训练仍为 100 轮、每客户端每轮 5 个完整 epoch，逻辑 batch=1024，物理 batch=128，最终轮及每 20 轮保存 checkpoint。下游保持最大 1000 epoch、每 10 epoch 验证、patience=15 次验证事件，验证同分保留首次。微调阶段关闭测试；全部模型选定并固定哈希后，再做独立最终测试。此次仅准备配置与代码，未启动这些任务或最终测试。

## 验证证据

- 全量回归 110 项通过，覆盖原 pipeline、batching、checkpoint、目标加速及方差正则等。
- 随后补充门控组合器的完整逻辑 batch 梯度等价检查，以及独立聚合入口的 checkpoint 重放和旧计数缺失处理；相关 37 项全部通过。
- 三组均通过两轮合成预训练、LoRA 微调、最佳模型保存与重载、基座冻结、训练期间不开测试的端到端检查。
- 18 份配置均对真实已准备数据通过路径、源目标角色及配置预检。
- 三组分别对 Cora 留一的五个真实源客户端各做一次逻辑 batch=4、物理 batch=2 的优化更新，再聚合、保存重载，并对 Cora 两个训练样本做一次 LoRA 优化；损失有限、两处 LoRA 均更新、基座冻结。
- 同时核对三组的 full/5-shot/1-shot 支持集记录一致。上述极小 batch 仅作代码验证，未运行正式消融轮次。
- 语法编译及 git diff --check 通过。远端当前无可见 CUDA 设备，验证环境为 CPU、PyTorch 2.6.0+cu124；GPU 数值与资源占用尚未验证。

机器可读证据位于工作区 `runs/ablation_pipeline_review_20261114/`：`full_tests.log`、`final_targeted_tests.log`、`real_data_validation.json`、`validation_summary.json`、`code_sha256.json` 和 `core_changes.patch`。18 份配置和矩阵计划位于 `planned_configs/`。该目录受 runs 忽略规则保护，不会随代码提交。

## 仅生成计划的入口

```bash
python -m fedg2c.node_ablation_matrix \
  --config config/fedg2c_nodes_ablation_20261114.json \
  --data-root /home/chenyh26/FedG2C/runs/gefra_loo_v1_data \
  --write-configs runs/ablation_plan_review
```

这个入口只解析和写出配置，没有执行训练的分支。用户审核通过后才同步 GitHub 与本地；正式启动实验仍需后续指令。
