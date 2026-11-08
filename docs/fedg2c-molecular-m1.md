# M1：分子图 CPU 工程验证

本协议检查真实分子输入上的 A→B→C 流程，不作为完整 OGB 基准成绩或组合泛化证据。

本次远端环境：Python 3.11.11、PyTorch 2.6.0+cu124、PyG 2.6.1、OGB 1.3.6；主机无可用 GPU，训练与评估均显式使用 CPU。

| 内容 | 固定设置 |
|---|---|
| 源客户端 1 | OGB-MolPCBA 官方 scaffold train 内固定抽取 10,000 个有效图 |
| 源客户端 2、3 | PROTEINS 的全部 1,113 图；IMDB-BINARY 的全部 1,000 图 |
| 目标数据 | OGB-MolHIV 官方 scaffold train/valid/test 内各抽取 4,096/1,024/1,024 图 |
| 子集选择 | seed=2026；目标在各官方分区内按原始类别比例固定抽样；不重新打乱官方分区 |
| 分子特征 | OGB 公布的 9 列原子类别、3 列键类别词表，使用私有嵌入适配器 |
| 其他图特征 | PROTEINS 缓存的 3 维特征；IMDB 节点度除以该图最大节点度 |
| 模型 | hidden_dim=64，4 专家，2 层 mysage；源视图上限 32 节点 |
| 聚合 | response，2 轮，各客户端每轮最多 3 个优化步骤；服务器使用已有 SBM 探针配置 |
| 下游 | none/lora/adapter/prompt；每种 2 epochs，batch_size=8，local_alpha=0，整图上限 256 |
| 评估 | 验证损失选模型，选定后测试一次；报告 ROC-AUC、AP 和基座哈希校验 |

源图文件不含标签或划分掩码，PCBA 每 256 图一个分片。采样允许的 PCBA 图必须至少有两个节点和一条非自环边，以满足遮边预测输入要求。记录实际过滤数、源子集索引、目标子集索引和源分片哈希。

本次数据导出共检查 10,001 个 PCBA 候选，跳过 1 个无效图后保留 10,000 个。三个源分别导出 40/5/4 个分片。MolHIV 子集包含如下标签分布，缺失标签均为 0；最大图有 200 个节点，使用完整图，不裁剪。

| 目标划分 | 图数 | 正例 | 负例 |
|---|---:|---:|---:|
| train | 4,096 | 153 | 3,943 |
| valid | 1,024 | 20 | 1,004 |
| test | 1,024 | 32 | 992 |

MolHIV 标签保持 0/1，缺失 -1 转成 NaN，已有 NaN 保留。当前官方 MolHIV 缓存无缺失标签；PCBA 的源标签不参与预训练。类别词表直接取 OGB 定义，不从目标验证/测试数据估计。不会裁剪目标图来规避尺寸限制。

联邦入口按清单顺序遍历图；工程预算每轮读取各客户端清单中的前三个图，各生成一个视图。两轮重复这些图，但使用不同的增广种子，即九个不同源图、18 次本地优化，不会遍历所有已准备图。10,000 个 PCBA 图是可用源子集，不代表每轮都经过优化。完整训练需另行增加预算并设计每轮采样，不应把本检查称为全量预训练。源/目标分子跨数据集的同分子去重、scaffold 隔离和组合因素留出尚未审计；M1 不提供这类泛化保证。服务器探针不含化学边属性。

运行方式：

```bash
PYTHON_BIN=/usr/bin/python bash scripts/fedg2c/run_molecular_cpu.sh
```

可设置 `RUN_ROOT`、`DATA_ROOT`、`CACHE_ROOT`、`SEED`、`ROUNDS`、`LOCAL_STEPS`、`FINETUNE_EPOCHS` 和 `ADAPTATIONS`。复用 `DATA_ROOT` 时应保持数据种子不变；新的训练运行使用不存在的 `RUN_ROOT`。脚本明确指定 CPU 并在导入 Python 之前限制库线程数。

复用已完成的数据导出时，脚本会检查种子和协议编号是否匹配：

```bash
PYTHON_BIN=/usr/bin/python \
DATA_ROOT="$PWD/runs/molecular_m1_data_20261022/data" \
RUN_ROOT="$PWD/runs/molecular_m1_cpu_new" \
bash scripts/fedg2c/run_molecular_cpu.sh
```

数据目录包含 `source_manifest.json`、`target_manifest.json` 和 `preparation_report.json`。训练目录保存 `console.log`、`federated/round_000`、`federated/round_001`，以及每种适配方式的 `target_<mode>/metrics.json`、`run_metadata.json`、`target_model.pt`。所有数据和训练产物均位于 Git 忽略的 `runs/` 下。

转换器为 PyG 切片创建独立张量存储，防止 `torch.save` 将完整数据集的底层存储重复写入每个分片。回归检查见 `tests/test_molecular_data.py`。

```bash
python -m unittest discover -s tests -p test_molecular_data.py -v
```

2026-10-22 的 M1 运行已完成。训练目录为 `runs/molecular_m1_cpu_20261022`，复用的数据目录为 `runs/molecular_m1_data_20261022/data`。四种适配方式各执行了 2 个 epoch，并由最低验证损失选中第 1 个 epoch；测试结果如下：

| 适配方式 | 测试 ROC-AUC | 测试 AP | 测试 BCE | 可训练参数 |
|---|---:|---:|---:|---:|
| none | 0.650674 | 0.059952 | 0.138318 | 12,033 |
| lora | 0.541205 | 0.033011 | 0.139733 | 13,057 |
| adapter | 0.664945 | 0.063552 | 0.136058 | 13,129 |
| prompt | 0.616841 | 0.045535 | 0.139875 | 12,097 |

四项回归检查通过。运行后的完整核验也通过：源图不含标签，所有源分片哈希匹配；PCBA 子集全部属于官方 train；目标三个子集互不重叠，均属于各自官方分区；6,144 个目标图的特征、边、键属性和标签与原始缓存逐图一致。六次客户端训练各有三个有限损失/梯度步骤，两个服务器轮次各执行十步 response 聚合。四个目标模型都能严格重载，并在验证图上输出有限预测；四组共享基座哈希均保持不变，来源均为同一个第 2 轮聚合检查点。

汇总保存在训练目录的 `verification_report.json`，完整指标保存在各 `target_<mode>/metrics.json`。该单种子、小预算检查只说明流程运行正常；表中差异不用于判断适配方法优劣。
