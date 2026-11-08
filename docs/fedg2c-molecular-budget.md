# 分子图预算审计与统一配置

2026-10-24 迁移到 `chenyh26@192.168.5.17:32230`。仓库仍为 `/home/chenyh26/FedG2C`，V100-SXM2 32GB、数据软链接及 `runs/molecular_formal_gpu_20261023` 完整保留。历史正式实验不被覆盖。

用户确认后续继续按最低验证 BCE 选模型、首次并列优先；模块 B 保留当前未归一化目标及权重，先比较每轮 10/20 步。超参数搜索关闭测试集评估。下述设置是工程基线与待检验候选，尚不是经过搜索的最优超参数。

## 已有结果能说明什么

### 模块 A：不能宣称 10 轮已经饱和

当前 10 轮、每客户端每轮 1,000 步、每图一个随机局部视图，共 30,000 步。PCBA 的 10,000 图刚好各访问一次；PROTEINS 访问 8–9 次，IMDB 访问 10 次。每轮的完整数据遍历比例分别为 0.1、约 0.898、1。一个图的一次访问只生成最多 32 节点的随机视图，也不代表看遍了整个图。

| 客户端 | 第 1 轮平均训练损失 | 第 10 轮平均训练损失 |
|---|---:|---:|
| MolPCBA | 0.04805 | 0.01416 |
| PROTEINS | 0.03539 | 0.04452 |
| IMDB-BINARY | 0.02840 | 0.05539 |

每轮图、增广、动量参考目标和共享状态都在变化，以上数值不是固定验证集上的收敛曲线。PCBA 仍在下降，另外两个客户端波动；不能据此断言饱和或表示塌缩。现有流程没有固定的源域验证集，判断更长预训练的收益需要固定下游验证协议，以及后续的多种子实验。

旧入口中 `local_epochs=1` 搭配 `shuffled_cycle + local_steps=1000` 是对选中的图流运行一次，不能称为完整客户端数据的一个 epoch。统一配置明确区分：

```json
"local_budget": {"unit": "steps", "amount": 1000}
```

表示每客户端每通信轮**恰好 1,000 次优化**。每步使用一个视图，`views_per_graph` 控制同一图的连续视图数。跨轮按确定性随机排列循环取图。

```json
"local_budget": {"unit": "epochs", "amount": 2}
```

表示每客户端每轮完整遍历自己的数据两次，每次独立打乱；优化步数为 `图数 × 2 × views_per_graph`。此模式会给 PCBA 更多优化步数，属于不同的计算分配协议，不能直接与等步数基线混为一谈。当前队列保持等步数。

本地 AdamW 在每轮重建，下发后动量参考分支重置为主分支。这是既定实现，不是跨通信轮次保留优化器或 EMA 状态的实验。

### 模块 B：10 步是合理基线，20 步值得对照，50 步暂不优先

使用历史每轮相同的三个上传权重、相同校准/诊断探针，分别独立重放 0/5/10/20/50 步；0 表示均匀权重初始化。CPU 重放不更新原实验，不使用目标数据。新主机结果按十轮取均值：

| 服务器步数 | 校准总目标 | 独立诊断输出 MSE | 独立诊断响应 MSE |
|---|---:|---:|---:|
| 0 | 0.108495 | 0.105385 | 3.125e-6 |
| 5 | 0.101935 | 0.100587 | 2.575e-6 |
| 10 | 0.099904 | 0.098673 | 2.686e-6 |
| 20 | 0.098132 | 0.096059 | 约 2.56e-6 |
| 50 | 0.097248 | 0.095986 | 约 2.56e-6 |

10→20 步的平均校准总目标下降约 1.77%，诊断输出 MSE 下降约 2.65%；20→50 步分别约 0.90% 和 0.077%。某些单轮诊断误差会变差。重放仅说明**固定上传条件下**的权重优化，不能代替改变服务器步数后整条联邦轨迹与下游表现的比较；诊断探针继续只用于报告，不挑选服务器权重或检查点。

另一个关键观察是尺度：10 步时，响应项对聚合 logits 的梯度范数与输出项的比值，按轮计算后平均约 0.00114，即 0.11%。输出/响应系数同为 1，并不意味着影响相当。这一测量只针对保存上传权重上的条件重放，且范数比受输出梯度接近零影响，不能直接视为两个模块的全局贡献占比。

响应已经按 `2 * epsilon` 计算中心有限差分；小尺度不是漏除扰动幅度。用户选择先保留公式与权重。后续可另设响应权重 1/100/1000/10000 或归一化目标的设计实验，不能把改公式与增加优化步数混在一起归因。

### 阶段二：20 epochs 够得到基线，但不足以证明排名指标饱和

四种适配的最低验证 BCE 分别出现在 none=18、lora=14、adapter=11、prompt=11 epochs。none 接近原上限；验证 ROC-AUC/AP 的最佳 epoch 也不完全等于 BCE 最佳 epoch。验证 BCE 在后半程波动，增加预算可能只延长过拟合，不能默认越久越好。

因此先比较固定基座下 `max_epochs=50, patience=10`：连续十个 epoch 验证 BCE 没有严格改善则停止，回到最低 BCE 检查点；并列保留较早模型。每轮同时报告验证 ROC-AUC/AP。若仍在上限附近改善，再增加到 100。该实验不读取或评估测试张量。

## 参数的设置依据与下一步搜索

原正式预算选择 10 通信轮、每轮 1,000 步、服务器 10 步、微调 20 epochs，是将 M1 扩展为可完成单种子正式训练的工程预算，**没有用充分的收敛实验确定**。不能给出“10 轮足够”的结论。

| 参数 | 当前基线 | 设置理由与建议 |
|---|---|---|
| 通信轮次 | 10 | PCBA 仅一个源数据遍历；先比较 20，必要时 40/80，以固定下游验证结果判断边际收益 |
| 每轮本地预算 | 1,000 步，1 视图/图 | 固定各客户端计算量；比较本地步数时建议每客户端总计 20,000 步：80×250、40×500、20×1000，避免把计算量增加误认为通信频率收益 |
| A 学习率/正则 | AdamW 1e-3 / 1e-5 | 常用初始量级，未搜索；预算确定后可比较 3e-4/1e-3/3e-3，以及 0/1e-5/1e-4 |
| A EMA/增广 | .99；遮边 .5，通道 .2 | 自监督工程起点，未做本项目最优性验证；后续分别检查 EMA .99/.995/.999、遮边 .3/.5/.7 |
| A 结构/视图 | 64 维，4 专家，2 层，视图 32，2-hop | 控制原型计算和容量；当前轮次比较全部固定，不同时改变容量 |
| B 优化 | 10 步，Adam lr=.05 | 三客户端仅优化聚合权重 logits，重放表明 20 步值得比较；先不增加到 50 |
| B 探针 | 校准 4、诊断 2；SBM [8,8]、概率 .5/.1，16 对读出、2 方向 | 小计算基线，估计方差未充分检验；后续校准 8/16、方向 4/8、多探针种子；诊断继续不能选权重 |
| B 响应/先验 | epsilon=.01，输出/响应/先验=1/1/.001 | 当前协议固定；尺度问题单列实验，不以 1:1 为“已经平衡”的证据 |
| 微调 epochs | 20 | 先增至最多 50、patience=10；选择依据仍为验证 BCE |
| 微调学习率/批量/正则 | AdamW 1e-3、batch=8、wd=0 | 小样本起点；先检查预算，再比较 3e-4/1e-3/3e-3 与 wd=0/1e-4，固定种子及数据 |
| 适配容量 | LoRA rank/alpha=4/4，adapter=8，local_alpha=0 | 轻量适配基线；轮次比较保持固定，后续再做 rank=2/4/8 与瓶颈=4/8/16 |
| 梯度范数上限 | A/C 均为 5 | 稳定训练的默认值，现已写入统一配置；未按结果挑选 |

预训练轮次变化首先改变源训练量；比较“每轮本地训练多少”则应控制总优化步数。多种子至少 3 个才能初步估计变动，当前队列仅种子 2026。还需无预训练共享基座对照，以及 fedavg/output_only/response 消融，才能讨论预训练和响应保持的实际收益。

## 一个配置运行两个阶段

新入口 `run_fedg2c_pipeline.py` 只读取指定 JSON，不用环境变量覆盖训练超参数；设备可见性仍由 CUDA 环境控制。路径相对仓库根目录解析，数据抽样种子 `data.prepare.seed` 与训练种子 `run.seed` 分开。换训练种子不应重抽一份数据。

`config/fedg2c_molecular_pipeline.json` 是原正式预算的完整配置：10×1000、B=10、四种适配各 20 epochs，最终测试开启。它是复现模板，本次没有重新运行该基线。

```bash
# 查看全部生效值和执行计划；不训练，不创建输出。
python run_fedg2c_pipeline.py --config config/fedg2c_molecular_pipeline.json --dry-run

# 完整两阶段：A+B 联邦预训练 → 最后一轮基座 → 四种独立下游适配。
python -u run_fedg2c_pipeline.py --config config/fedg2c_molecular_pipeline.json

# 仅阶段一。
python -u run_fedg2c_pipeline.py --config config/fedg2c_molecular_pipeline.json --stage stage1

# 复用已有基座，仅做验证集上的微调预算实验。
python -u run_fedg2c_pipeline.py --config config/fedg2c_molecular_target_budget.json --stage stage2
```

每次使用新的 `run.output`。运行已存在时直接报错；不隐式恢复或覆盖。`--stage stage2` 要求在同一配置中指定 `stage2.checkpoint`；完整运行时该字段必须为 null，明确使用本次阶段一产物。

配置组织：

| 区块 | 控制内容 |
|---|---|
| run | 输出目录、训练种子、设备、CPU 线程 |
| data | 源/目标 manifest；可选 M1 准备协议、缓存、数据输出、抽样种子 |
| stage1.training | 通信轮次、本地 steps/epochs、每图视图数、日志频率 |
| stage1.module_a | 模型结构、视图/遮蔽、EMA、本地优化器参数、梯度上限 |
| stage1.module_b | 聚合模式、服务器迭代/学习率、探针、目标各项权重 |
| stage2 | 可选已有基座、适配方式列表；target 内含学习率、epoch 上限、选模指标、早停、是否测试等 |

未准备数据时可按既定 M1 协议导出；已存在则复用并核对准备种子和协议 ID，不重新下载。M1 的数据数量仍是已确认数据协议中的常量，改变数量需要另外设计协议。未知字段与不兼容预算会报错。`resolved_config.json` 包含默认补全后的实际参数，弥补原脚本用 CLI 将 2 epochs 覆盖成 20 的不直观问题。

输出含 `stage1/round_*/`、`stage2/{mode}/`、`summary.json`、`console.log`、`status.json`、`exit_code`、输入/生效配置、执行计划、环境、代码快照与哈希、manifest 哈希。目标检查点仍可独立重载，冻结基座检查继续执行。`evaluate_test=false` 时仅打开 train/valid 张量，对 test 只检查 manifest 身份信息；最终 `test=null, test_evaluated=false`。

## 本次队列

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/fedg2c/run_molecular_budget_suite.py \
  --output runs/molecular_budget_suite_20261024
```

队列在启动前检查三份配置，保存冻结的配置副本，按顺序执行；任一运行失败即停，不启动后续任务。关闭 SSH 后由已启动的 nohup 任务继续执行。状态在队列的 `status.json`，每个任务同时保存自己的状态与日志。

| 顺序 | 配置 | 内容 | 输出 |
|---|---|---|---|
| 1 | fedg2c_molecular_target_budget.json | 原第 10 轮基座，四种适配最多 50 epochs，patience=10 | runs/molecular_target_budget_20261024 |
| 2 | fedg2c_molecular_round20_server10.json | 20×1000 预训练，B=10，再做相同微调 | runs/molecular_round20_server10_20261024 |
| 3 | fedg2c_molecular_round20_server20.json | 20×1000 预训练，B=20，其余与任务 2 相同 | runs/molecular_round20_server20_20261024 |

三份配置都保持数据 M1、seed=2026、A lr=1e-3/epochs 语义、B 原始目标、四种适配、微调 lr=1e-3、验证 BCE 首次最小选模；不读取测试张量。20 轮预训练产生的中间每轮基座也保留，供后续固定协议检查轮次收益。先得出这些结果，再安排等总步数本地预算、学习率及多种子实验。

`scripts/fedg2c/compare_molecular_budgets.py runs/molecular_budget_suite_20261024 --wait` 可等待队列完成，再生成 `comparison.json` 与 `comparison.md`，按验证 BCE 选择的模型对比原基线与三组新实验。新主机另有后台进程执行该汇总；不再次运行测试。队列失败时汇总也停止并报错。

这不是全超参数网格搜索，尚不能称上述候选“已优于基线”。原基线测试集已经评估过，本次调参不能让它重新成为完全未见的测试集；最终结果仍需单独协议与多种子报告。

## 复核与关键代码

新主机 CPU 预算重放文件为 `runs/molecular_budget_analysis_20261024/server_replay.json`，已同步本地。汇总与图像可由以下命令重建，读取目标验证历史而不读取测试指标：

```bash
python scripts/fedg2c/summarize_molecular_budget.py runs/molecular_formal_gpu_20261023 \
  --replay runs/molecular_budget_analysis_20261024/server_replay.json \
  --output runs/molecular_budget_analysis_20261024/budget_summary.json --plot
```

| 代码 | 责任 |
|---|---|
| fedg2c/pipeline.py | 严格解析与补全配置、阶段连接、快照、日志和失败状态 |
| pretrain_federated_fedg2c.py | A/B 联邦轮次；固定步数循环取图与完整 shuffled_epoch |
| fedg2c/edge_client.py | 逐视图优化、梯度裁剪、每步更新 EMA |
| fedg2c/response_server.py | 保持原目标的服务器权重优化 |
| fedg2c/target_train.py | 每轮验证、首个最佳模型、早停、可关闭的最终测试 |
| fedg2c/target_data.py | 验证模式不加载 test 张量、清单身份/划分检查 |
| scripts/fedg2c/audit_server_budget.py | 保存上传与探针的条件重放、两项梯度范数 |

本地 6 项新入口检查通过；新主机初轮全部 12 项检查通过，环境变量 `FEDG2C_TEST_DEVICE=cuda` 使两阶段整合检查在 GPU 上执行。补充“默认协议仅在选模后测试一次”的检查后共 13 项。检查包含固定步数、完整客户端 epoch、全部适配的基座重载、验证选模方向/并列/早停、未定义指标报错、测试文件不存在时验证模式仍可运行，以及原 7 项数据与采样检查。接口检查是代码验证，本次正式超参数比较是独立队列。
