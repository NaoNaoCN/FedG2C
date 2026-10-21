# 模块 B 的公开代码审阅快照

固定来源、提交、URL 与哈希见 [manifest.json](manifest.json)。这些文件保留原样，仅供审阅，不在主程序中导入，也没有加 A/B/C 区块注释。

- FED-PUB（ICML 2023）：审阅 `Server.get_proxy_data` 的随机分块图和标准高斯特征、`Client.get_functional_embedding` 的功能表示。固定提交未见根目录许可证；本项目未复制其算法代码进入运行实现，也不对快照重新授权。
- FedLAW（ICML 2023）：审阅 `fedlaw_optimization` 的权重参数化、可微参数组合和监督目标。MIT 许可证保留在其目录。
- GAD（ICML 2026）：审阅 `train_gad_distillation.py` 的归一化方向、无梯度参考和有限差分响应。MIT 许可证保留在其目录。

运行实现直接调用 PyG 的 `stochastic_blockmodel_graph`、PyTorch 的 `functional_call` 和优化器；图关系门控、节点对读出和无标签响应损失是本项目适配。详见 [模块 B 设计](../../02_服务器聚合_关系响应保持.md)。
