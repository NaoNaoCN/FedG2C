# 三组消融部署（2026-11-14）

用户已审核通过核心实现并授权部署训练。主机 `chenyh26@192.168.5.2:31214`，Tesla V100-SXM2-32GB，容器内存约 24 GiB，8 个 CPU 配额。使用 `/usr/bin/python`，PyTorch 2.6.0+cu124。部署工作区 `/home/chenyh26/FedG2C_worktrees/ablation_deploy_20261114`，分支 `codex/ablation-deploy-20261114`。

正式输出 `/home/chenyh26/FedG2C/runs/nodes_ablation_20261114`。冻结代码、18 份配置、输入数据哈希和资源检查证据保存在该目录。复用历史准备数据 `/home/chenyh26/FedG2C/runs/gefra_loo_v1_data`；不复用历史训练权重。

## 实验和调度

沿用已批准的三个设置 baseline、ablate_b、ablate_ab，六个节点分类目标，18 次预训练和 540 次 LoRA 微调。每个设置和目标独立训练 100 轮，每轮各源客户端 5 个完整 epoch，逻辑 batch=1024、物理 batch=128。私有输入适配器 bias 归零和 both/reference 方差正则在三组统一启用。下游 seeds 0–9，full/5-shot/1-shot 支持集逐项核对；保留历史模型初始化顺序。

专用调度器 `scripts/fedg2c/run_node_ablation_suite.py` 依次执行三个阶段：完成全部预训练；完成全部验证集选模；固定 540 个选定 checkpoint 的哈希和支持集后，独立运行最终测试。验证同分保留第一次，测试结果不参与选模。结果写入 `RESULTS.md`、`summary/report.json`，完整性审计写入 `summary/audit.json`。

并发上限为两个预训练进程、一个下游进程。启动前检查 GPU 与可用内存，运行中监测内存和显存；发生资源终止时降到单进程并有限重试。预训练 GPU 启动门槛为剩余 10000 MiB；下游为剩余 26000 MiB；可用内存至少 6 GiB。低于剩余 1800 MiB 显存或内存工作集超过限额 86% 时终止最后一个 worker 并从恢复点重试。

## 恢复和代码隔离

预训练 `--recover` 保存所有客户端在线/EMA 模型、私有参数、当前共享状态、轮次和客户端游标、已完成上传、CPU/CUDA/Python 随机数状态。保存边界为初始化、每个完整客户端训练后和每轮服务器聚合后。客户端内部被中断时，重做这个客户端；原训练每次客户端调用均重建优化器，因而无额外优化器跨调用状态需要保留。

下游 `run_target(..., recover=True)` 在验证事件后原子保存模型、AdamW、shuffle generator、CPU/CUDA/Python 随机状态、最佳模型和早停计数。中断后回到最近验证事件继续；训练数据、共享参数哈希或配置改变会拒绝恢复。恢复点和最终选模模型分别保存，最终模型不用于优化器恢复。

每个任务检查冻结代码哈希，所有训练来自 `code_snapshot/`，避免后续 checkout 编辑影响正在运行的任务。文件锁及继承到 worker 的锁描述符防止重复启动同一套实验。中断发生在首个恢复点之前时，未完成目录移到 `interrupted_attempts/` 后重新执行，保留原记录。

## 验证

部署前全套测试 126 项通过，覆盖三组预训练和微调、私有状态恢复与连续运行逐张量一致、断点恢复选模一致、阶段屏障、支持集和 checkpoint 哈希审计、测试重复执行跳过。GPU 核心回归 73 项通过。源端正式逻辑 batch=1024/物理 batch=128 在三组的 CiteSeer 和 WikiCS 真实数据上各执行一次完整优化更新；目标端对两种组合器的六个目标执行正式 microbatch 的 LoRA 更新，损失有限、两处 LoRA 均更新、基座保持冻结。

GPU 执行检查中源端峰值 reserved 约 2.35 GiB；目标单次 microbatch 峰值 reserved 约 1.90 GiB。这是单步检查，正式流程仍使用保守并发和持续资源监测。检查输出不是正式 checkpoint。

证据位于部署工作区 `runs/deployment_preflight_20261114/`；正式目录的 `preflight/` 保存其副本。运行状态由 `status.json` 和 `logs/supervisor.log` 判断。

## 启动和恢复

首次准备和验证证据录入后，从冻结代码启动：

```bash
/usr/bin/python -u /home/chenyh26/FedG2C/runs/nodes_ablation_20261114/code_snapshot/scripts/fedg2c/run_node_ablation_suite.py run --root /home/chenyh26/FedG2C/runs/nodes_ablation_20261114
```

部署使用脱离 SSH 的进程组，日志重定向并记录 PID。退出后使用相同命令从恢复点续跑；正常运行期间重复启动会被文件锁拒绝。此部署不安装开机自启服务。
